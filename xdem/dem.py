"""DEM class and functions."""
from __future__ import annotations

import os
import pathlib
import warnings
from typing import Any, Literal

import pyproj
import rasterio as rio

from geoutils import SatelliteImage
from geoutils.raster import RasterType
from pyproj import Transformer, CRS
from pyproj.crs import BoundCRS, VerticalCRS, CompoundCRS, GeographicCRS
from pyproj.crs.coordinate_system import Ellipsoidal3DCS
from pyproj.crs.enums import Ellipsoidal3DCSAxis

from xdem._typing import NDArrayf


def _parse_vcrs_name_from_product(product: str) -> str | None:
    """
    Parse vertical CRS name from DEM product name.

    :param product: Product name (typically from satimg.parse_metadata_from_fn).

    :return: vcrs_name: Vertical CRS name.
    """
    # Sources for defining vertical references:
    # AW3D30: https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v11_format_e.pdf
    # SRTMGL1: https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf
    # SRTMv4.1: http://www.cgiar-csi.org/data/srtm-90m-digital-elevation-database-v4-1
    # ASTGTM2/ASTGTM3: https://lpdaac.usgs.gov/documents/434/ASTGTM_User_Guide_V3.pdf
    # NASADEM: https://lpdaac.usgs.gov/documents/592/NASADEM_User_Guide_V1.pdf, HGTS is ellipsoid, HGT is EGM96 geoid !!
    # ArcticDEM (mosaic and strips): https://www.pgc.umn.edu/data/arcticdem/
    # REMA (mosaic and strips): https://www.pgc.umn.edu/data/rema/
    # TanDEM-X 90m global: https://geoservice.dlr.de/web/dataguide/tdm90/
    # COPERNICUS DEM: https://spacedata.copernicus.eu/web/cscda/dataset-details?articleId=394198

    if product in ["ArcticDEM/REMA", "TDM1", "NASADEM-HGTS"]:
        vcrs_name = "WGS84"
    elif product in ["AW3D30", "SRTMv4.1", "SRTMGL1", "ASTGTM2", "NASADEM-HGT"]:
        vcrs_name = "EGM96"
    elif product in ["COPDEM"]:
        vcrs_name = "EGM08"
    else:
        vcrs_name = None

    return vcrs_name


def _build_ccrs_from_crs_and_vcrs(crs: CRS, vcrs: VerticalCRS | Literal["Ellipsoid"]) -> CompoundCRS | CRS:
    """
    Build a compound CRS from a horizontal CRS and a vertical CRS.

    :param crs: Horizontal CRS.
    :param vcrs: Vertical CRS.

    :return: Compound CRS (horizontal + vertical).
    """

    # If a vertical CRS was passed, build a compound CRS with horizontal + vertical
    # This requires transforming the horizontal CRS to 2D in case it was 3D
    # Using CRS() because rasterio.CRS does not allow to call .name otherwise...
    if isinstance(vcrs, CRS):
        ccrs = CompoundCRS(
            name="Horizontal: " + CRS(crs).name + "; Vertical: " + vcrs.name,
            components=[CRS(crs).to_2d(), vcrs],
        )
    # Else if "Ellipsoid" was passed, there is no vertical reference
    # We still have to return the CRS in 3D
    else:
        ccrs = CRS(crs).to_3d()

    return ccrs


def _build_vcrs_from_grid(grid: str | pathlib.Path, old_way: bool = True) -> CompoundCRS:
    """
    Build a compound CRS from a vertical CRS grid path.

    :param grid: Path to grid for vertical reference.
    :param old_way: Whether to use the new or old way of building the compound CRS with pyproj (for testing purposes).

    :return: Compound CRS (horizontal + vertical).
    """

    if not os.path.exists(os.path.join(pyproj.datadir.get_data_dir(), grid)):
        raise ValueError("Grid not found in " + str(pyproj.datadir.get_data_dir()) + ": check if proj-data is "
                         "installed via conda-forge, the pyproj.datadir, and that you are using a grid available at "
                         "https://github.com/OSGeo/PROJ-data."
                         )

    # The old way: see https://gis.stackexchange.com/questions/352277/.
    if old_way:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            ccrs = pyproj.Proj(init="EPSG:4326",
                               geoidgrids=grid).crs
            bound_crs = ccrs.sub_crs_list[1]

    # The clean way (to respect the new PROJ indexing order?)
    else:
        # First, we build a bounds CRS (the vertical CRS relative to geographic)
        vertical_crs = VerticalCRS(name="unknown", datum='VDATUM["unknown"]')
        geographic3d_crs = GeographicCRS(
            name="WGS 84",
            ellipsoidal_cs=Ellipsoidal3DCS(
                axis=Ellipsoidal3DCSAxis.LATITUDE_LONGITUDE_HEIGHT
            ),
        )
        bound_crs = BoundCRS(
            source_crs=vertical_crs,
            target_crs=geographic3d_crs,
            transformation={
                "$schema": "https://proj.org/schemas/v0.2/projjson.schema.json",
                "type": "Transformation",
                "name": "unknown to WGS84 ellipsoidal height",
                "source_crs": vertical_crs.to_json_dict(),
                "target_crs": geographic3d_crs.to_json_dict(),
                "method": {
                    "name": "GravityRelatedHeight to Geographic3D"
                },
                "parameters": [
                    {
                        "name": "Geoid (height correction) model file",
                        "value": os.path.exists(os.path.join(pyproj.datadir.get_data_dir(), grid)),
                    }
                ]
            }
        )

    return bound_crs


# Define CRS in case path or string was passed
_vcrs_meta_from_name = {"EGM08": {"grid": "us_nga_egm08_25.tif", "epsg": 3855},  # EGM2008 at 2.5 minute resolution
                        "EGM96": {"grid": "us_nga_egm96_15.tif", "epsg": 5773}}  # EGM1996 at 15 minute resolution


def _vcrs_from_user_input(
        new_vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | CRS | int
        ) -> VerticalCRS | Literal["Ellipsoid"]:
    """
    Parse vertical CRS from user input.

    :param new_vcrs: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
        a EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).

    :return: Vertical CRS.
    """

    # Raise errors if input type is wrong (allow CRS instead of VerticalCRS for broader error messages below)
    if not isinstance(new_vcrs, (str, pathlib.Path, CRS, int)):
        raise TypeError("New vertical CRS must be a name or path as string or "
                        "a vertical reference as pyproj.crs.VerticalCRS, received {}.".format(type(new_vcrs)))

    # If input is ellipsoid
    if isinstance(new_vcrs, str) and (new_vcrs.lower() == "ellipsoid" or new_vcrs.upper() == 'WGS84'):
        return "Ellipsoid"

    # Define CRS in case EPSG or CRS was passed
    if isinstance(new_vcrs, (int | CRS)):
        if isinstance(new_vcrs, int):
            vcrs = CRS.from_epsg(new_vcrs)
        else:
            vcrs = new_vcrs

        # Raise errors if the CRS constructed is not vertical or has other components
        if isinstance(new_vcrs, CRS) and not new_vcrs.is_vertical:
            raise ValueError("New vertical CRS must have a vertical axis (check with is_vertical).")
        elif isinstance(new_vcrs, CRS) and not isinstance(new_vcrs, VerticalCRS) and new_vcrs.is_vertical:
            warnings.warn("New vertical CRS has a vertical dimension but also other components, "
                          "extracting the first vertical reference only.")
            vcrs = [subcrs for subcrs in new_vcrs.sub_crs_list if subcrs.is_vertical][0]

    # If a string was passed
    else:
        # If a name is passed, define CRS based on dict
        if isinstance(new_vcrs, str) and new_vcrs.upper() in _vcrs_meta_from_name.keys():
            vcrs_meta = _vcrs_meta_from_name[new_vcrs]
            vcrs = CRS.from_epsg(vcrs_meta["epsg"])
        # Otherwise, attempt to read a grid from the string
        else:
            if isinstance(new_vcrs, pathlib.Path):
                grid = new_vcrs.name
            else:
                grid = new_vcrs
            vcrs = _build_vcrs_from_grid(grid=grid)

    return vcrs


dem_attrs = ["_vcrs", "_vcrs_name", "_vcrs_grid", "_ccrs"]


class DEM(SatelliteImage):  # type: ignore
    """
    The digital elevation model.

    The DEM has a single additional main attribute to that inherited from :class:`geoutils.SatelliteImage`
    and :class:`geoutils.Raster`:
        vcrs: :class:`pyproj.VerticalCRS`
            Vertical coordinate reference system of the DEM.

    Other derivative attributes are:
        vcrs_name: :class:`str`
            Name of vertical CRS of the DEM.
        vcrs_grid: :class:`str`
            Grid path to the vertical CRS of the DEM.
        ccrs: :class:`pyproj.CompoundCRS`
            Compound vertical and horizontal CRS of the DEM.

    The DEM also inherits from :class:`geoutils.Raster`:
        data: :class:`np.ndarray`
            Data array of the DEM, with dimensions corresponding to (count, height, width).
        transform: :class:`affine.Affine`
            Geotransform of the DEM.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the DEM.
        nodata: :class:`int` or :class:`float`
            Nodata value of the DEM.

    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    def __init__(
        self,
        filename_or_dataset: str | RasterType | rio.io.DatasetReader | rio.io.MemoryFile,
        vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | VerticalCRS | str | pathlib.Path | int | None = None,
        silent: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Instantiate a digital elevation model.

        The vertical reference of the DEM can be defined by passing the `vcrs` argument.
        Otherwise, a vertical reference is tentatively parsed from the DEM product name.

        Inherits all attributes from the :class:`geoutils.Raster` and :class:`geoutils.SatelliteImage` classes.

        :param filename_or_dataset: The filename of the dataset.
        :param vcrs: Vertical coordinate reference system either as a name ("WGS84", "EGM08", "EGM96"),
            a EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        :param silent: Whether to display vertical reference parsing.
        """

        self.data: NDArrayf
        self._vcrs: VerticalCRS | None = None
        self._vcrs_name: str | None = None
        self._vcrs_grid: str | None = None
        self._ccrs: CompoundCRS | None = None

        # If DEM is passed, simply point back to DEM
        if isinstance(filename_or_dataset, DEM):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent SatelliteImage class options (including raised errors)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Parse metadata from file not implemented")
                super().__init__(filename_or_dataset, silent=silent, **kwargs)

        # Ensure DEM has only one band: self.indexes can be None when data is not loaded through the Raster class
        if self.indexes is not None and len(self.indexes) > 1:
            raise ValueError("DEM rasters should be composed of one band only")

        # If no vertical CRS was provided by the user
        if vcrs is None:
            vcrs = _parse_vcrs_name_from_product(self.product)

        # If a vertical reference was parsed or provided by user
        if vcrs is not None:
            self.set_vcrs(vcrs)

    def copy(self, new_array: NDArrayf | None = None) -> DEM:
        """
        Copy the DEM, possibly updating the data array.

        :param new_array: New data array.

        :return: Copied DEM.
        """

        new_dem = super().copy(new_array=new_array)  # type: ignore
        # The rest of attributes are immutable, including pyproj.CRS
        # dem_attrs = ['vref','vref_grid','ccrs'] #taken outside of class
        for attrs in dem_attrs:
            setattr(new_dem, attrs, getattr(self, attrs))

        return new_dem  # type: ignore

    @property
    def vcrs(self) -> VerticalCRS | Literal["Ellipsoid"] | None:
        """Vertical coordinate reference system of the DEM."""

        return self._vcrs

    @property
    def vcrs_grid(self) -> str | None:
        """Grid path of vertical coordinate reference system of the DEM."""

        return self._vcrs_grid

    @property
    def vcrs_name(self) -> str | None:
        """Name of vertical coordinate reference system of the DEM."""

        if self.vcrs is not None:
            # If it is the ellipsoid
            if isinstance(self.vcrs, str):
                # Need to call CRS() here to make it work with rasterio.CRS...
                vcrs_name = "Ellipsoid (No vertical CRS). Datum: {}.".format(CRS(self.crs).ellipsoid.name)
            # Otherwise, return the vertical reference name
            else:
                vcrs_name = self.vcrs.name
        else:
            vcrs_name = None

        return vcrs_name

    def set_vcrs(self, new_vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | VerticalCRS | int) -> None:
        """
        Set the vertical coordinate reference system of the DEM.

        :param new_vcrs: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
            a EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        """

        # Get vertical CRS and set it
        vcrs = _vcrs_from_user_input(new_vcrs=new_vcrs)
        self._vcrs = vcrs

        # If a grid or name was passed, also set vcrs_grid
        if isinstance(new_vcrs, (str, pathlib.Path)):
            # If the string is within the supported names
            if isinstance(new_vcrs, str) and new_vcrs in _vcrs_meta_from_name.keys():
                grid = _vcrs_meta_from_name[new_vcrs]["grid"]
            # If it's a pathlib path
            elif isinstance(new_vcrs, pathlib.Path):
                grid = new_vcrs.name
            # Or an ellipsoid
            elif new_vcrs.lower() == "ellipsoid":
                grid = None
            # Or a string path
            else:
                grid = new_vcrs

            self._vcrs_grid = grid
        # Otherwise, explicitly set the new vcrs_grid to None
        else:
            self._vcrs_grid = None

    @property
    def ccrs(self) -> CompoundCRS | CRS | None:
        """Compound horizontal and vertical coordinate reference system of the DEM."""

        if self.vcrs is not None:
            ccrs = _build_ccrs_from_crs_and_vcrs(crs=self.crs, vcrs=self.vcrs)
            return ccrs
        else:
            return None

    def to_vcrs(self,
                dst_vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
                src_vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None = None) -> None:
        """
        Convert the DEM to another vertical coordinate reference system.

        :param dst_vcrs: Destination vertical CRS. Either as a name ("WGS84", "EGM08", "EGM96"),
            a EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data)
        :param src_vcrs: Force a source vertical CRS (uses metadata by default). Same formats as for `dst_vcrs`.

        :return:
        """

        if self.vcrs is None and src_vcrs is None:
            raise ValueError(
                "The current DEM has no vertical reference, define one with .set_vref() or by passing `src_vcrs` to perform a conversion."
            )

        # Initial Compound CRS (only exists if vertical CRS is not None, as checked above)
        if src_vcrs is not None:
            # Warn if a vertical CRS already existed for that DEM
            if self.vcrs is not None:
                warnings.warn(category=UserWarning, message="Overriding the vertical CRS of the DEM with the one provided in `src_vcrs`.")
            ccrs_init = _build_ccrs_from_crs_and_vcrs(self.crs, vcrs=src_vcrs)
        else:
            ccrs_init = self.ccrs

        # New destination Compound CRS
        ccrs_dest = _build_ccrs_from_crs_and_vcrs(self.crs, vcrs=_vcrs_from_user_input(new_vcrs=dst_vcrs))

        # Transform the grid
        transformer = Transformer.from_crs(crs_from=ccrs_init, crs_to=ccrs_dest, always_xy=True)
        # Will preserve the mask of the masked-array since pyproj 3.4
        zz = self.data
        xx, yy = self.coords(offset="center")
        zz_trans = transformer.transform(xx, yy, zz)[2]

        # Update DEM
        self.data = zz_trans.astype(self.dtypes[0])

        # Update vcrs (which will update ccrs if called)
        self.set_vcrs(new_vcrs=dst_vcrs)

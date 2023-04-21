"""DEM class and functions."""
from __future__ import annotations

import pathlib
import warnings
from typing import Any, Literal

import rasterio as rio
from affine import Affine
from geoutils import SatelliteImage
from geoutils.raster import RasterType
from pyproj import CRS
from pyproj.crs import CompoundCRS, VerticalCRS

from xdem._typing import MArrayf, NDArrayf
from xdem.vcrs import (
    _build_ccrs_from_crs_and_vcrs,
    _grid_from_user_input,
    _parse_vcrs_name_from_product,
    _transform_zz,
    _vcrs_from_crs,
    _vcrs_from_user_input,
)

dem_attrs = ["_vcrs", "_vcrs_name", "_vcrs_grid"]


class DEM(SatelliteImage):  # type: ignore
    """
    The digital elevation model.

    The DEM has a single main attribute in addition to that inherited from :class:`geoutils.Raster`:
        vcrs: :class:`pyproj.VerticalCRS`
            Vertical coordinate reference system of the DEM.

    Other derivative attributes are:
        vcrs_name: :class:`str`
            Name of vertical CRS of the DEM.
        vcrs_grid: :class:`str`
            Grid path to the vertical CRS of the DEM.
        ccrs: :class:`pyproj.CompoundCRS`
            Compound vertical and horizontal CRS of the DEM.

    The attributes inherited from :class:`geoutils.Raster` are:
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
        vcrs: Literal["Ellipsoid"]
        | Literal["EGM08"]
        | Literal["EGM96"]
        | VerticalCRS
        | str
        | pathlib.Path
        | int
        | None = None,
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
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        :param silent: Whether to display vertical reference parsing.
        """

        self.data: NDArrayf
        self._vcrs: VerticalCRS | Literal["Ellipsoid"] | None = None
        self._vcrs_name: str | None = None
        self._vcrs_grid: str | None = None

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

        # If the CRS in the raster metadata has a 3rd dimension, could set it as a vertical reference
        vcrs_from_crs = _vcrs_from_crs(CRS(self.crs))
        if vcrs_from_crs is not None:
            # If something was also provided by the user, user takes precedence
            # (we leave vcrs as it was for input)
            if vcrs is not None:
                # Raise a warning if the two are not the same
                vcrs_user = _vcrs_from_user_input(vcrs)
                if not vcrs_from_crs == vcrs_user:
                    warnings.warn(
                        "The CRS in the raster metadata already has a vertical component, "
                        "the user-input '{}' will override it.".format(vcrs)
                    )
            # Otherwise, use the one from the raster 3D CRS
            else:
                vcrs = vcrs_from_crs

        # If no vertical CRS was provided by the user or defined in the CRS
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
        for attrs in dem_attrs:
            setattr(new_dem, attrs, getattr(self, attrs))

        return new_dem  # type: ignore

    @classmethod
    def from_array(
        cls: type[DEM],
        data: NDArrayf | MArrayf,
        transform: tuple[float, ...] | Affine,
        crs: CRS | int | None,
        nodata: int | float | None = None,
        vcrs: Literal["Ellipsoid"]
        | Literal["EGM08"]
        | Literal["EGM96"]
        | str
        | pathlib.Path
        | VerticalCRS
        | int
        | None = None,
    ) -> DEM:
        """Create a DEM from a numpy array and the georeferencing information.

        :param data: Input array.
        :param transform: Affine 2D transform. Either a tuple(x_res, 0.0, top_left_x,
            0.0, y_res, top_left_y) or an affine.Affine object.
        :param crs: Coordinate reference system. Either a rasterio CRS,
            or an EPSG integer.
        :param nodata: Nodata value.
        :param vcrs: Vertical coordinate reference system.

        :returns: DEM created from the provided array and georeferencing.
        """
        # We first apply the from_array of the parent class
        rast = SatelliteImage.from_array(data=data, transform=transform, crs=crs, nodata=nodata)
        # Then add the vcrs to the class call (that builds on top of the parent class)
        return cls(filename_or_dataset=rast, vcrs=vcrs)

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
                vcrs_name = f"Ellipsoid (No vertical CRS). Datum: {CRS(self.crs).ellipsoid.name}."
            # Otherwise, return the vertical reference name
            else:
                vcrs_name = self.vcrs.name
        else:
            vcrs_name = None

        return vcrs_name

    def set_vcrs(
        self,
        new_vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | VerticalCRS | int,
    ) -> None:
        """
        Set the vertical coordinate reference system of the DEM.

        :param new_vcrs: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        """

        # Get vertical CRS and set it and the grid
        self._vcrs = _vcrs_from_user_input(vcrs_input=new_vcrs)
        self._vcrs_grid = _grid_from_user_input(vcrs_input=new_vcrs)

    @property
    def ccrs(self) -> CompoundCRS | CRS | None:
        """Compound horizontal and vertical coordinate reference system of the DEM."""

        if self.vcrs is not None:
            ccrs = _build_ccrs_from_crs_and_vcrs(crs=self.crs, vcrs=self.vcrs)
            return ccrs
        else:
            return None

    def to_vcrs(
        self,
        dst_vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        src_vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None = None,
    ) -> None:
        """
        Convert the DEM to another vertical coordinate reference system.

        :param dst_vcrs: Destination vertical CRS. Either as a name ("WGS84", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data)
        :param src_vcrs: Force a source vertical CRS (uses metadata by default). Same formats as for `dst_vcrs`.

        :return:
        """

        if self.vcrs is None and src_vcrs is None:
            raise ValueError(
                "The current DEM has no vertical reference, define one with .set_vref() "
                "or by passing `src_vcrs` to perform a conversion."
            )

        # Initial Compound CRS (only exists if vertical CRS is not None, as checked above)
        if src_vcrs is not None:
            # Warn if a vertical CRS already existed for that DEM
            if self.vcrs is not None:
                warnings.warn(
                    category=UserWarning,
                    message="Overriding the vertical CRS of the DEM with the one provided in `src_vcrs`.",
                )
            src_ccrs = _build_ccrs_from_crs_and_vcrs(self.crs, vcrs=src_vcrs)
        else:
            src_ccrs = self.ccrs

        # New destination Compound CRS
        dst_ccrs = _build_ccrs_from_crs_and_vcrs(self.crs, vcrs=_vcrs_from_user_input(vcrs_input=dst_vcrs))

        # If both compound CCRS are equal, do not run any transform
        if src_ccrs.equals(dst_ccrs):
            warnings.warn(
                message="Source and destination vertical CRS are the same, skipping vertical transformation.",
                category=UserWarning,
            )
            return None

        # Transform elevation with new vertical CRS
        zz = self.data
        xx, yy = self.coords(offset="center")
        zz_trans = _transform_zz(crs_from=src_ccrs, crs_to=dst_ccrs, xx=xx, yy=yy, zz=zz)

        # Update DEM
        self._data = zz_trans.astype(self.dtypes[0])  # type: ignore

        # Update vcrs (which will update ccrs if called)
        self.set_vcrs(new_vcrs=dst_vcrs)

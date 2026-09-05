# Copyright (c) 2026 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Routines for vertical CRS transformation (fully based on pyproj)."""

from __future__ import annotations

import os
import pathlib
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from urllib.error import HTTPError

import affine
import numpy as np
import pyproj
from geoutils._dispatch import get_geo_attr
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.mparray import map_overlap
from geoutils.raster.referencing import _ij2xy
from pyproj import CRS
from pyproj.crs import BoundCRS, CompoundCRS, GeographicCRS, VerticalCRS
from pyproj.crs.coordinate_system import Ellipsoidal3DCS
from pyproj.crs.enums import Ellipsoidal3DCSAxis
from pyproj.transformer import TransformerGroup

from xdem._misc import import_optional
from xdem._typing import MArrayf, NDArrayf

if TYPE_CHECKING:
    import dask.array as da

    from xdem import DEM
    from xdem.dem.base import DEMBase, DEMLike

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
vcrs_dem_products = {
    "ArcticDEM/REMA/EarthDEM": "Ellipsoid",
    "TDM1": "Ellipsoid",
    "NASADEM-HGTS": "Ellipsoid",
    "AW3D30": "EGM96",
    "SRTMv4.1": "EGM96",
    "ASTGTM2": "EGM96",
    "ASTGTM3": "EGM96",
    "NASADEM-HGT": "EGM96",
    "COPDEM": "EGM08",
}


def _check_vcrs_input(vcrs: Any, crs: Any) -> Any:
    """
    Process user-input vertical CRS and CRS, and return normalized CRS output.

    :param vcrs: Vertical CRS input.
    :param crs: CRS input.

    :return: Normalized CRS output.
    """

    # Parse 2D/3D CRS
    if crs is None:
        if vcrs is not None:
            raise ValueError("A horizontal CRS is required before setting a vertical CRS.")
        return None
    crs = pyproj.CRS.from_user_input(crs)

    # Vertical CRS from different sources
    vcrs_from_crs = _vcrs_from_crs(crs)
    if vcrs is None:
        vcrs_from_user = None
    else:
        vcrs_from_user = _vcrs_from_user_input(vcrs)

    # Determine which vertical CRS to use
    if vcrs_from_user is not None:
        # User input takes precedence over CRS metadata
        if vcrs_from_crs is not None and vcrs_from_user != vcrs_from_crs:
            warnings.warn(
                "The CRS in the elevation metadata already has a vertical component, "
                f"the user-provided '{vcrs}' will override it."
            )
        out_vcrs = vcrs_from_user
    else:
        out_vcrs = vcrs_from_crs

    # Build final CRS
    if out_vcrs is not None:
        out_crs = _combine_crs_and_vcrs(crs, out_vcrs)
    else:
        out_crs = crs

    return out_crs


# EPSG codes for units
_UNIT_SYMBOLS = {
    "9001": "m",  # metre
    "9002": "ft",  # foot
    "9003": "ftUS",  # US survey foot
    "9036": "km",
    "9102": "°",
}


def vertical_unit_symbol(crs: Any) -> str | None:
    """
    Return the short unit symbol of the vertical axis (e.g. "m", "ft").

    Returns None if the CRS has no vertical axis.
    """

    # Process CRS input
    crs = CRS(crs)

    # If compound CRS, isolate the vertical CRS
    if crs.is_compound:
        crs = crs.sub_crs_list[1]

    # Check axis is indeed vertical, otherwise return None
    for axis in crs.axis_info:
        if axis.direction in ("up", "down"):

            # Prefer EPSG unit code if it exists
            code = axis.unit_auth_code
            if code and code in _UNIT_SYMBOLS:
                return _UNIT_SYMBOLS[code]

            # Fallback to normalized unit names
            name = axis.unit_name.lower()

            if name in {"metre", "meter"}:
                return "m"

            if name == "kilometre":
                return "km"

            if name == "foot":
                return "ft"

            if name == "us survey foot":
                return "ftUS"

            return axis.unit_name

    return None


def _parse_vcrs_name_from_product(product: str) -> str | None:
    """
    Parse vertical CRS name from DEM product name.

    :param product: Product name (typically from satimg.parse_metadata_from_fn).

    :return: vcrs_name: Vertical CRS name.
    """

    if product in vcrs_dem_products.keys():
        vcrs_name = vcrs_dem_products[product]
    else:
        vcrs_name = None

    return vcrs_name


def _combine_crs_and_vcrs(crs: CRS, vcrs: CRS | Literal["Ellipsoid"]) -> CRS:
    """
    Build a 3D CRS (compound or expanded) from a horizontal CRS and a vertical CRS input.

    :param crs: Horizontal CRS.
    :param vcrs: Vertical CRS.

    :return: 3D CRS (horizontal + vertical).
    """

    # If a vertical CRS was passed, build a compound CRS with horizontal + vertical
    # This requires transforming the horizontal CRS to 2D in case it was 3D
    # Using CRS() because rasterio.CRS does not allow to call .name otherwise...
    if isinstance(vcrs, CRS):
        # If pyproj >= 3.5.1, we can use CRS.to_2d()
        from packaging.version import Version

        if Version(pyproj.__version__) >= Version("3.5.1"):
            crs_from = CRS(crs).to_2d()
            combined_crs = CompoundCRS(
                name="Horizontal: " + CRS(crs).name + "; Vertical: " + vcrs.name,
                components=[crs_from, vcrs],
            )
        # Otherwise, we have to raise an error if the horizontal CRS is already 3D
        else:
            crs_from = CRS(crs)
            # If 3D
            if len(crs_from.axis_info) > 2:
                raise NotImplementedError(
                    "pyproj >= 3.5.1 is required to demote a 3D CRS to 2D and be able to compound "
                    "with a new vertical CRS. Update your dependencies or pass the 2D source CRS "
                    "manually."
                )
            # If 2D
            else:
                combined_crs = CompoundCRS(
                    name="Horizontal: " + CRS(crs).name + "; Vertical: " + vcrs.name,
                    components=[crs_from, vcrs],
                )

    # Else if "Ellipsoid" was passed, there is no vertical CRS, but we expand the ellipsoid to 3D
    # We isolate the 2D horizontal CRS (removing potential geoids), then expand it to 3D
    elif isinstance(vcrs, str) and vcrs.lower() == "ellipsoid":
        combined_crs = CRS(crs).to_2d().to_3d()
    else:
        raise ValueError("Invalid vcrs given. Must be a vertical CRS or the literal string 'Ellipsoid'.")

    return combined_crs


def _build_vcrs_from_grid(grid: str, old_way: bool = False) -> BoundCRS:
    """
    Build a bound CRS from a vertical CRS grid path.

    :param grid: Path to grid for vertical reference.
    :param old_way: Whether to use the new or old way of building the compound CRS with pyproj (for testing purposes).

    :return: Bound CRS.
    """

    if not os.path.exists(os.path.join(pyproj.datadir.get_data_dir(), grid)):
        warnings.warn(
            f"Grid '{grid}' not found in {pyproj.datadir.get_data_dir()}. Attempting to download from "
            f"https://cdn.proj.org/..."
        )
        from pyproj.sync import _download_resource_file

        try:
            _download_resource_file(
                file_url=os.path.join("https://cdn.proj.org/", grid),
                short_name=grid,
                directory=pyproj.datadir.get_data_dir(),
                verbose=False,
            )
        except HTTPError:
            raise ValueError(
                "The provided grid '{}' does not exist at https://cdn.proj.org/. "
                "Provide an existing grid.".format(grid)
            )

    # The old way: see https://gis.stackexchange.com/questions/352277/.
    if old_way:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            combined_crs = pyproj.Proj(init="EPSG:4326", geoidgrids=grid).crs
            bound_crs = combined_crs.sub_crs_list[1]

    # The clean way
    else:
        # First, we build a bounds CRS (the vertical CRS relative to geographic)
        vertical_crs = VerticalCRS(
            name="unknown using geoidgrids=" + grid, datum='VDATUM["unknown using geoidgrids=' + grid + '"]'
        )
        geographic3d_crs = GeographicCRS(
            name="WGS 84",
            ellipsoidal_cs=Ellipsoidal3DCS(axis=Ellipsoidal3DCSAxis.LATITUDE_LONGITUDE_HEIGHT),
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
                "method": {"name": "GravityRelatedHeight to Geographic3D"},
                "parameters": [
                    {
                        "name": "Geoid (height correction) model file",
                        "value": grid,
                        "id": {"authority": "EPSG", "code": 8666},
                    }
                ],
            },
        )

    return bound_crs


# Define types of common Vertical CRS dictionary
class VCRSMetaDict(TypedDict, total=False):
    grid: str
    epsg: int


_vcrs_meta: dict[str, VCRSMetaDict] = {
    "EGM08": {"grid": "us_nga_egm08_25.tif", "epsg": 3855},  # EGM2008 at 2.5 minute resolution
    "EGM96": {"grid": "us_nga_egm96_15.tif", "epsg": 5773},  # EGM1996 at 15 minute resolution
}


def _vcrs_from_crs(crs: CRS | None) -> CRS | Literal["Ellipsoid"] | None:
    """Get the vertical CRS from a CRS."""

    # If no CRS is defined
    if crs is None:
        return None
    else:
        crs = CRS(crs)

    # Check if CRS is 3D
    if len(crs.axis_info) > 2:

        # Check if CRS has a vertical compound
        if any(subcrs.is_vertical for subcrs in crs.sub_crs_list):
            # Then we get the first vertical CRS (should be only one anyway)
            vcrs = [subcrs for subcrs in crs.sub_crs_list if subcrs.is_vertical][0]
        # Otherwise, it's a 3D CRS based on an ellipsoid
        else:
            vcrs = "Ellipsoid"
    # Otherwise, the CRS is 2D and there is no vertical CRS
    else:
        vcrs = None

    return vcrs


def _vcrs_from_user_input(
    vcrs_input: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | CRS | int,
) -> VerticalCRS | BoundCRS | Literal["Ellipsoid"]:
    """
    Parse vertical CRS from user input.

    :param vcrs_input: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
        an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).

    :return: Vertical CRS.
    """

    # Raise errors if input type is wrong (allow CRS instead of VerticalCRS for broader error messages below)
    if not isinstance(vcrs_input, (str, pathlib.Path, CRS, int)):
        raise TypeError(f"New vertical CRS must be a string, path or VerticalCRS, received {type(vcrs_input)}.")

    # If input is ellipsoid
    if (
        (isinstance(vcrs_input, str) and (vcrs_input.lower() == "ellipsoid" or vcrs_input.upper() == "WGS84"))
        or (isinstance(vcrs_input, int) and vcrs_input in [4326, 4979])
        or (isinstance(vcrs_input, CRS) and vcrs_input.to_epsg() in [4326, 4979])
    ):
        return "Ellipsoid"

    # Define CRS in case EPSG or CRS was passed
    if isinstance(vcrs_input, (int, CRS)):
        if isinstance(vcrs_input, int):
            vcrs = CRS.from_epsg(vcrs_input)
        else:
            vcrs = vcrs_input

        # Raise errors if the CRS constructed is not vertical or has other components
        if isinstance(vcrs, CRS) and not vcrs.is_vertical:
            raise ValueError(
                "New vertical CRS must have a vertical axis, '{}' does not "
                "(check with `CRS.is_vertical`).".format(vcrs.name)
            )
        elif isinstance(vcrs, CRS) and vcrs.is_vertical and len(vcrs.axis_info) > 2:
            warnings.warn(
                "New vertical CRS has a vertical dimension but also other components, "
                "extracting the vertical reference only."
            )
            vcrs = _vcrs_from_crs(vcrs)

    # If a string or path was passed
    else:
        if isinstance(vcrs_input, pathlib.Path):
            vcrs_input = vcrs_input.name
        # If a name is passed, define CRS based on dict
        key = vcrs_input.upper()
        if isinstance(vcrs_input, str) and key in _vcrs_meta:
            vcrs_meta = _vcrs_meta[key]
            vcrs = CRS.from_epsg(vcrs_meta["epsg"])
        # Otherwise, attempt to read a grid from the string
        elif os.path.splitext(vcrs_input)[-1].lower() in [".tif", ".json", ".pol"]:
            if isinstance(vcrs_input, pathlib.Path):
                grid = vcrs_input.name
            else:
                grid = vcrs_input
            vcrs = _build_vcrs_from_grid(grid=grid)
        else:
            all_keys = ", ".join(_vcrs_meta.keys()) + ", Ellipsoid"
            raise ValueError(
                f"String vcrs input '{vcrs_input}' is not recognized. Must be one of '"
                f"{all_keys}' or a path with extension .tif/.json/.pol to a PROJ grid file."
            )

    return vcrs


def _build_vertical_transformer(crs_from: CRS, crs_to: CRS) -> pyproj.Transformer:
    """
    Build the best available transformer for a vertical CRS transformation.

    Downloads missing grids before returning, if needed.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Best transformation is not available")
        trans_group = TransformerGroup(crs_from=crs_from, crs_to=crs_to, always_xy=True)

    # Download grid if best available is not on disk, then re-initialize
    if not trans_group.best_available:
        trans_group.download_grids()
        trans_group = TransformerGroup(crs_from=crs_from, crs_to=crs_to, always_xy=True)

    # If the best available grid is still not there, raise a warning
    if not trans_group.best_available:
        warnings.warn(
            category=UserWarning,
            message="Best available grid for transformation could not be downloaded, "
            "applying the next best available (caution: might apply no transform at all).",
        )

    return trans_group.transformers[0]


def _transform_zz(
    transformer: pyproj.Transformer,
    xx: NDArrayf,
    yy: NDArrayf,
    zz: MArrayf | NDArrayf | int | float,
) -> MArrayf | NDArrayf | int | float:
    """
    Transform elevation to a new 3D CRS using an already-built transformer.
    """

    # Will preserve the mask of the masked-array since pyproj 3.4
    zz_trans = transformer.transform(xx, yy, zz)[2]

    return zz_trans


# Vertical CRS transformation for DEMs
######################################


def _to_vcrs_2d_pyproj(
    data: NDArrayf,
    transform: affine.Affine,
    transformer: pyproj.Transformer,
    area_or_point: Literal["Area", "Point"] | None = None,
    pixel_offset: tuple[int, int] = (0, 0),
) -> NDArrayf:
    """
    Transform raster elevations at their original pixel coordinates, optionally starting at a tile's row and column.

    We need the original transform and global pixel indices because rebuilding coordinates from a tile's origin
    changes floating-point rounding for fractional pixel sizes. Each backend must sample the geoid at the same points.
    """

    # 1/ Prepare the global pixel indices, preserving an optional band axis
    has_band_axis = data.ndim == 3 and data.shape[0] == 1
    if has_band_axis:
        data = data[0]
    row_start, col_start = pixel_offset
    rows = np.arange(row_start, row_start + data.shape[0])
    cols = np.arange(col_start, col_start + data.shape[1])

    # 2/ Transform elevations at the same coordinates as the full raster
    # GeoUtils applies the shared Area/Point convention before the original affine transform
    xx, _ = _ij2xy(i=0, j=cols, transform=transform, area_or_point=area_or_point)
    _, yy = _ij2xy(i=rows, j=0, transform=transform, area_or_point=area_or_point)
    xx, yy = np.meshgrid(xx, yy)
    zz_trans = _transform_zz(
        transformer=transformer,
        xx=xx,
        yy=yy,
        zz=data,
    )

    # 3/ Retain the input mask, floating precision and dimensions after transformation
    values = np.asanyarray(zz_trans).astype(data.dtype, copy=False)
    return values[None, ...] if has_band_axis else values


def _to_vcrs_2d_block_dask(
    data: NDArrayf,
    *,
    transform: affine.Affine,
    src_crs_wkt: str,
    dst_crs_wkt: str,
    area_or_point: Literal["Area", "Point"] | None = None,
    block_info: list[dict[str, Any]] | None = None,
) -> NDArrayf:
    """Transform a Dask block at global pixel indices supplied by block_info."""

    if block_info is None:
        raise ValueError("block_info must be provided.")

    # Read the tile's starting indices without recomputing a rounded affine origin
    row_loc, col_loc = block_info[0]["array-location"][-2:]

    # Dask may return slices or (start, stop) tuples depending on version
    row_start = row_loc.start if hasattr(row_loc, "start") else row_loc[0]
    col_start = col_loc.start if hasattr(col_loc, "start") else col_loc[0]

    # Rebuild transformer inside the block (serialization issues with a Pyproj transformer if passing it)
    transformer = _build_vertical_transformer(
        crs_from=CRS.from_wkt(src_crs_wkt),
        crs_to=CRS.from_wkt(dst_crs_wkt),
    )

    return _to_vcrs_2d_pyproj(
        data=data,
        transform=transform,
        transformer=transformer,
        area_or_point=area_or_point,
        pixel_offset=(row_start, col_start),
    )


def _dask_to_vcrs_2d(
    darr: da.Array,
    transform: affine.Affine,
    src_crs: CRS,
    dst_crs: CRS,
    area_or_point: Literal["Area", "Point"] | None = None,
) -> da.Array:
    """Blockwise vertical CRS transform using Dask."""

    # Simply use map_blocks, as all transformations are independent when purely vertical
    import_optional("dask")
    return darr.map_blocks(
        _to_vcrs_2d_block_dask,
        transform=transform,
        src_crs_wkt=src_crs.to_wkt(),
        dst_crs_wkt=dst_crs.to_wkt(),
        area_or_point=area_or_point,
        dtype=darr.dtype,
        meta=np.array((), dtype=darr.dtype),
    )


def _to_vcrs_2d_block_mp(
    dem: DEM,
    src_crs_wkt: str,
    dst_crs_wkt: str,
    source_transform: affine.Affine,
) -> DEM:
    """Transform a multiprocessing tile using the original raster transform and global pixel indices."""

    # Rebuild transformer inside the block (serialization issues with a Pyproj transformer if passing it)
    transformer = _build_vertical_transformer(
        crs_from=CRS.from_wkt(src_crs_wkt),
        crs_to=CRS.from_wkt(dst_crs_wkt),
    )

    # Worker windows lie on the source grid; round inverse coordinates to recover their integer offsets
    col_start, row_start = ~source_transform * (dem.transform.c, dem.transform.f)
    pixel_offset = (int(round(row_start)), int(round(col_start)))

    # Transform using global indices so fractional pixels match the eager and Dask calculations exactly
    out_data = _to_vcrs_2d_pyproj(
        data=dem.data,
        transform=source_transform,
        transformer=transformer,
        area_or_point=dem.area_or_point,
        pixel_offset=pixel_offset,
    )

    return dem.from_array(
        data=out_data,
        transform=dem.transform,
        crs=CRS.from_wkt(dst_crs_wkt),
        nodata=dem.nodata,
        area_or_point=dem.area_or_point,
        tags=dem.tags,
    )


def _multiproc_to_vcrs_2d(
    dem: DEM,
    *,
    src_crs: CRS,
    dst_crs: CRS,
    mp_config: MultiprocConfig,
) -> DEM:
    """
    Vertical CRS transform using multiprocessing.
    """

    out_dem = map_overlap(
        _to_vcrs_2d_block_mp,
        dem,
        mp_config,
        src_crs.to_wkt(),
        dst_crs.to_wkt(),
        source_transform=dem.transform,
        depth=0,
    )

    # GeoUtils' overlap writer retains the input grid CRS, so update the file's vertical metadata explicitly
    import rasterio as rio

    from xdem.dem.dem import DEM

    with rio.open(out_dem.name, "r+") as dataset:
        dataset.crs = dst_crs
        dataset.update_tags(**dem.tags)
    return DEM(out_dem.name)


def _get_vertical_transform_crss(
    crs: Any,
    dst_vcrs: Any,
    force_source_vcrs: Any | None = None,
) -> tuple[CRS, CRS]:
    """
    Build source and destination 3D CRS for a vertical transformation, and raise errors where necessary.
    """

    if crs is None:
        raise ValueError("A horizontal CRS is required before transforming a vertical CRS.")

    # Get source VCRS from current CRS
    src_vcrs = _vcrs_from_crs(crs)

    # Early exit if conversion not defined
    if src_vcrs is None and force_source_vcrs is None:
        raise ValueError(
            "The elevation data have no vertical reference, define one with .set_vcrs() "
            "or by passing `force_source_vcrs` to perform a conversion."
        )

    # Build the source 3D CRS
    if force_source_vcrs is not None:
        if src_vcrs is not None:
            warnings.warn(
                category=UserWarning,
                message=f"Overriding the vertical CRS of the elevation data "
                f"with the one provided in `force_source_vcrs`: {force_source_vcrs}.",
            )
        force_src_vcrs = _vcrs_from_user_input(force_source_vcrs)
        src_crs = _combine_crs_and_vcrs(crs, vcrs=force_src_vcrs)
    else:
        src_crs = crs

    # Build the destination 3D CRS
    dst_crs = _combine_crs_and_vcrs(
        crs,
        vcrs=_vcrs_from_user_input(vcrs_input=dst_vcrs),
    )

    return src_crs, dst_crs


def _to_vcrs_2d(
    dem: DEMBase,
    dst_vcrs: Any,
    force_source_vcrs: Any | None = None,
    mp_config: MultiprocConfig | None = None,
) -> DEMLike | None:
    """
    Transform DEM to a different vertical CRS (no change in horizontal CRS).

    Supports direct in-memory execution, Dask execution, and Multiprocessing.

    :param dem: DEM.
    :param dst_vcrs: Destination vertical CRS.
    :param force_source_vcrs: Force the source vertical CRS if not defined or to override it.
    :param mp_config: Multiprocessing configuration.
    :returns: DEM or DataArray with transformed elevations and destination CRS, or None when no change is needed.
    """

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    dask_backend = dem._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from to_vcrs(). To use Multiprocessing, use a DEM object input and pass mp_config."
        )

    # Build source and destination 3D CRS from the input vertical CRSs
    src_crs, dst_crs = _get_vertical_transform_crss(
        crs=dem.crs,
        dst_vcrs=dst_vcrs,
        force_source_vcrs=force_source_vcrs,
    )
    transform = get_geo_attr(dem, "transform")

    # If both 3D CRS are equal, do not run any transform
    if src_crs.equals(dst_crs):
        warnings.warn(
            message="Source and destination vertical CRS are the same, skipping vertical transformation.",
            category=UserWarning,
        )
        return None

    # Build transformer once to trigger grid download outside of parallelization + validate best available transform
    # We won't be able to pass the transformer directly to the chunked functions (not serializable),
    # so we'll repass the src/dst CRS
    _build_vertical_transformer(crs_from=src_crs, crs_to=dst_crs)

    # Multiprocessing backend
    if mp_backend:
        dem_out = _multiproc_to_vcrs_2d(
            dem=dem,
            src_crs=src_crs,
            dst_crs=dst_crs,
            mp_config=mp_config,
        )
        return dem_out

    else:
        # Dask backend
        if dask_backend:
            zz_trans = _dask_to_vcrs_2d(
                darr=dem.data,
                transform=transform,
                src_crs=src_crs,
                dst_crs=dst_crs,
                area_or_point=dem.area_or_point,
            )
        else:
            # Direct NumPy backend
            transformer = _build_vertical_transformer(crs_from=src_crs, crs_to=dst_crs)
            zz_trans = _to_vcrs_2d_pyproj(
                data=dem.data,
                transform=transform,
                transformer=transformer,
                area_or_point=dem.area_or_point,
            )

        dem_out = dem.copy(new_array=zz_trans)
        get_geo_attr(dem_out, "set_crs")(dst_crs)

        return dem_out


# Shared vertical metadata for raster and point elevation data
############################################################


class _VerticalReference(ABC):
    """Keep VCRS metadata and its manipulation consistent across DEMBase and EPCBase.

    We need this because DEMs and elevation point clouds use the same vertical reference metadata and rules for
    reading or setting it. Keeping that code here avoids duplicating it in DEMBase and EPCBase, so their native
    classes and accessors all use the same implementation.

    The public ``vcrs`` property returns the vertical part. The existing ``crs`` property returns the complete 3D
    reference, so separate properties for its name, grid or combined form are unnecessary.

    Each concrete backend supplies CRS access and assignment without loading elevations. Numerical vertical
    transformations stay in the raster and point implementations because their data representations differ.
    """

    @property
    @abstractmethod
    def crs(self) -> Any:
        """Horizontal or three-dimensional CRS, without loading elevation values."""

    @abstractmethod
    def _set_vcrs_crs(self, new_crs: CRS) -> None:
        """Replace CRS metadata without loading elevation values."""

    @property
    def vcrs(self) -> VerticalCRS | Literal["Ellipsoid"] | None:
        """
        Vertical coordinate reference system of the elevation data.
        """
        return _vcrs_from_crs(self.crs)

    def set_vcrs(
        self,
        new_vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | VerticalCRS | int,
    ) -> None:
        """
        Set the vertical coordinate reference system of the elevation data.

        :param new_vcrs: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        """

        # Require horizontal referencing before combining it with the vertical datum
        if self.crs is None:
            raise ValueError("A horizontal CRS is required before setting a vertical CRS.")

        # Get vertical CRS and re-set the CRS
        new_vcrs = _vcrs_from_user_input(vcrs_input=new_vcrs)
        new_crs = _combine_crs_and_vcrs(crs=self.crs, vcrs=new_vcrs)
        self._set_vcrs_crs(new_crs)


# Vertical CRS transformation for elevation point clouds
########################################################


def _to_vcrs_1d_dataframe(ds: Any, data_column: str | None, src_crs: CRS, dst_crs: CRS) -> Any:
    """Transform one point partition, preserving its index, auxiliary columns and elevation dtype."""

    import geopandas as gpd

    # Build the transformer inside the worker and transform only elevation values
    transformer = _build_vertical_transformer(src_crs, dst_crs)
    elevations = ds[data_column].to_numpy() if data_column is not None else ds.geometry.z.to_numpy()
    transformed = _transform_zz(transformer, ds.geometry.x.to_numpy(), ds.geometry.y.to_numpy(), elevations)
    values = np.asarray(transformed).astype(elevations.dtype, copy=False)

    # Preserve horizontal geometry and auxiliary data while replacing the elevation representation
    result = ds.copy()
    if data_column is not None:
        result[data_column] = values
    else:
        result.geometry = gpd.points_from_xy(ds.geometry.x, ds.geometry.y, z=values, crs=ds.crs)
    result.set_crs(dst_crs, allow_override=True, inplace=True)
    result.attrs["data_column"] = data_column
    return result


def _to_vcrs_1d(
    epc: Any,
    dst_vcrs: Any,
    force_source_vcrs: Any = None,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """Transform native or accessor elevation points with eager, Dask or multiprocessing execution."""

    from geoutils.pointcloud.base import _get_dataframe_attrs, _set_dataframe_attrs
    from geoutils.pointcloud.las import _point_partition_size

    # Reject mixed backends before inspecting any elevation values
    if mp_config is not None and epc._is_dask:
        raise ValueError("Cannot use multiprocessing and Dask simultaneously. Remove mp_config for Dask inputs.")
    src_crs, dst_crs = _get_vertical_transform_crss(epc.crs, dst_vcrs, force_source_vcrs)
    if src_crs.equals(dst_crs):
        warnings.warn("Source and destination vertical CRS are the same, skipping vertical transformation.")
        return epc.copy() if epc.is_loaded or epc._is_pd else epc.__class__(epc)

    # Resolve any missing grid once before sending work to workers
    _build_vertical_transformer(src_crs, dst_crs)
    if epc._is_dask:
        if epc.data_column is None:
            raise ValueError("Dask-backed point clouds require an explicit data column.")
        meta = epc.ds._meta.set_crs(dst_crs, allow_override=True)
        result = epc.ds.map_partitions(_to_vcrs_1d_dataframe, epc.data_column, src_crs, dst_crs, meta=meta)
        attrs = _get_dataframe_attrs(epc.ds).copy()
        attrs["crs"] = dst_crs
        _set_dataframe_attrs(result, attrs)
        return result

    # Load a separate native object for multiprocessing so the source keeps its metadata-only state
    if mp_config is not None:
        import pandas as pd

        source = epc
        if not epc.is_loaded:
            source = epc.__class__(epc)
            source.load(mp_config=mp_config)
        partition_size = _point_partition_size(mp_config)
        futures = [
            mp_config.cluster.submit(
                _to_vcrs_1d_dataframe,
                source.ds.iloc[start : start + partition_size],
                epc.data_column,
                src_crs,
                dst_crs,
            )
            for start in range(0, source.point_count, partition_size)
        ]
        parts = mp_config.cluster.gather(futures)
        result = pd.concat(parts) if parts else _to_vcrs_1d_dataframe(source.ds, epc.data_column, src_crs, dst_crs)
    else:
        result = _to_vcrs_1d_dataframe(epc.ds, epc.data_column, src_crs, dst_crs)

    # Reconstruct the runtime native class or return the dataframe directly for accessors
    if epc._is_pd:
        return result
    return epc.__class__(result, data_column=epc.data_column)

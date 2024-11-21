# Copyright (c) 2024 xDEM developers
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

import http.client
import os
import pathlib
import warnings
from typing import Literal, TypedDict

import pyproj
from pyproj import CRS
from pyproj.crs import BoundCRS, CompoundCRS, GeographicCRS, VerticalCRS
from pyproj.crs.coordinate_system import Ellipsoidal3DCS
from pyproj.crs.enums import Ellipsoidal3DCSAxis
from pyproj.transformer import TransformerGroup

from xdem._typing import MArrayf, NDArrayf

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


def _build_ccrs_from_crs_and_vcrs(crs: CRS, vcrs: CRS | Literal["Ellipsoid"]) -> CompoundCRS | CRS:
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
        # If pyproj >= 3.5.1, we can use CRS.to_2d()
        from packaging.version import Version

        if Version(pyproj.__version__) > Version("3.5.0"):
            crs_from = CRS(crs).to_2d()
            ccrs = CompoundCRS(
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
                ccrs = CompoundCRS(
                    name="Horizontal: " + CRS(crs).name + "; Vertical: " + vcrs.name,
                    components=[crs_from, vcrs],
                )

    # Else if "Ellipsoid" was passed, there is no vertical reference
    # We still have to return the CRS in 3D
    elif isinstance(vcrs, str) and vcrs.lower() == "ellipsoid":
        ccrs = CRS(crs).to_3d()
    else:
        raise ValueError("Invalid vcrs given. Must be a vertical CRS or the literal string 'Ellipsoid'.")

    return ccrs


def _build_vcrs_from_grid(grid: str, old_way: bool = False) -> CompoundCRS:
    """
    Build a compound CRS from a vertical CRS grid path.

    :param grid: Path to grid for vertical reference.
    :param old_way: Whether to use the new or old way of building the compound CRS with pyproj (for testing purposes).

    :return: Compound CRS (horizontal + vertical).
    """

    if not os.path.exists(os.path.join(pyproj.datadir.get_data_dir(), grid)):
        warnings.warn(
            "Grid not found in "
            + str(pyproj.datadir.get_data_dir())
            + ". Attempting to download from https://cdn.proj.org/..."
        )
        from pyproj.sync import _download_resource_file

        try:
            _download_resource_file(
                file_url=os.path.join("https://cdn.proj.org/", grid),
                short_name=grid,
                directory=pyproj.datadir.get_data_dir(),
                verbose=False,
            )
        except http.client.InvalidURL:
            raise ValueError(
                "The provided grid '{}' does not exist at https://cdn.proj.org/. "
                "Provide an existing grid.".format(grid)
            )

    # The old way: see https://gis.stackexchange.com/questions/352277/.
    if old_way:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            ccrs = pyproj.Proj(init="EPSG:4326", geoidgrids=grid).crs
            bound_crs = ccrs.sub_crs_list[1]

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


def _vcrs_from_crs(crs: CRS) -> CRS:
    """Get the vertical CRS from a CRS."""

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

    # If a string was passed
    else:
        # If a name is passed, define CRS based on dict
        if isinstance(vcrs_input, str) and vcrs_input.upper() in _vcrs_meta.keys():
            vcrs_meta = _vcrs_meta[vcrs_input]
            vcrs = CRS.from_epsg(vcrs_meta["epsg"])
        # Otherwise, attempt to read a grid from the string
        else:
            if isinstance(vcrs_input, pathlib.Path):
                grid = vcrs_input.name
            else:
                grid = vcrs_input
            vcrs = _build_vcrs_from_grid(grid=grid)

    return vcrs


def _grid_from_user_input(vcrs_input: str | pathlib.Path | int | CRS) -> str | None:

    # If a grid or name was passed, get grid name
    if isinstance(vcrs_input, (str, pathlib.Path)):
        # If the string is within the supported names
        if isinstance(vcrs_input, str) and vcrs_input in _vcrs_meta.keys():
            grid = _vcrs_meta[vcrs_input]["grid"]
        # If it's a pathlib path
        elif isinstance(vcrs_input, pathlib.Path):
            grid = vcrs_input.name
        # Or an ellipsoid
        elif vcrs_input.lower() == "ellipsoid":
            grid = None
        # Or a string path
        else:
            grid = vcrs_input
    # Otherwise, there is none
    else:
        grid = None

    return grid


def _transform_zz(
    crs_from: CRS, crs_to: CRS, xx: NDArrayf, yy: NDArrayf, zz: MArrayf | NDArrayf | int | float
) -> MArrayf | NDArrayf | int | float:
    """
    Transform elevation to a new 3D CRS.

    :param crs_from: Source CRS.
    :param crs_to: Destination CRS.
    :param xx: X coordinates.
    :param yy: Y coordinates.
    :param zz: Z coordinates.

    :return: Transformed Z coordinates.
    """

    # Find all possible transforms
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Best transformation is not available")
        trans_group = TransformerGroup(crs_from=crs_from, crs_to=crs_to, always_xy=True)

    # Download grid if best available is not on disk, download and re-initiate the object
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
    transformer = trans_group.transformers[0]

    # Will preserve the mask of the masked-array since pyproj 3.4
    zz_trans = transformer.transform(xx, yy, zz)[2]

    return zz_trans

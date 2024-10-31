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

import logging
import os

import geoutils

from xdem import (  # noqa
    coreg,
    dem,
    examples,
    filters,
    fit,
    spatialstats,
    terrain,
    volume,
)
from xdem.coreg.workflows import dem_coregistration
from xdem.ddem import dDEM  # noqa
from xdem.dem import DEM  # noqa
from xdem.demcollection import DEMCollection  # noqa

try:
    from xdem._version import __version__  # noqa
except ImportError:  # pragma: no cover
    raise ImportError(
        "xDEM is not properly installed. If you are "
        "running from the source directory, please instead "
        "create a new virtual environment (using conda or "
        "virtualenv) and then install it in-place by running: "
        "pip install -e ."
    )


def coregister(ref_dem_path: str, tba_dem_path: str) -> None:
    """
    Function to compare and coregister Digital Elevation Models (DEMs).

    This function verifies the existence of the provided DEM paths,
    loads the reference DEM and the DEM to be aligned, and performs
    coregistration. The aligned DEM and an inlier mask are then saved
    to disk.

    :param ref_dem_path: Path to the reference DEM file.
    :param tba_dem_path: Path to the DEM that needs to be aligned to the reference.
    :return:
    :raises FileNotFoundError: if the reference DEM or the DEM to be aligned does not exist.
    """
    # Verify that both DEM paths exist
    if not os.path.exists(ref_dem_path):
        raise FileNotFoundError(f"Reference DEM path does not exist: {ref_dem_path}")
    if not os.path.exists(tba_dem_path):
        raise FileNotFoundError(f"DEM to be aligned path does not exist: {tba_dem_path}")

    logging.info("Loading DEMs: %s, %s", ref_dem_path, tba_dem_path)

    # Load the reference and secondary DEMs
    reference_dem, to_be_aligned_dem = geoutils.raster.load_multiple_rasters([ref_dem_path, tba_dem_path])

    # Execute coregistration
    logging.info("Starting coregistration...")
    coreg_dem, coreg_method, out_stats, inlier_mask = dem_coregistration(
        to_be_aligned_dem, reference_dem, "aligned_dem.tiff"
    )

    # Save outputs
    logging.info("Saving aligned DEM and inlier mask...")
    inlier_rst = coreg_dem.copy(new_array=inlier_mask)
    inlier_rst.save("inlier_mask.tiff")

    # Print the coregistration details
    print(coreg_method.info())
    print("Coregistration statistics:\n", out_stats)
    logging.info("Coregistration completed")

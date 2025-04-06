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

"""Utility functions to download and find example data."""
import os
import shutil
import tarfile
import tempfile
import urllib.request

import geoutils as gu

import xdem

_DATA_REPO_URL = "https://github.com/GlacioHack/xdem-data/tarball/main"
_COMMIT_HASH = "98004a09f84def4c78b253d41b212baca2b3cccb"

_EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "data"))
# Absolute filepaths to the example files.
_FILEPATHS_DATA = {
    "longyearbyen_ref_dem": os.path.join(_EXAMPLES_DIRECTORY, "Longyearbyen", "data", "DEM_2009_ref.tif"),
    "longyearbyen_tba_dem": os.path.join(_EXAMPLES_DIRECTORY, "Longyearbyen", "data", "DEM_1990.tif"),
    "longyearbyen_glacier_outlines": os.path.join(
        _EXAMPLES_DIRECTORY, "Longyearbyen", "data", "glacier_mask", "CryoClim_GAO_SJ_1990.shp"
    ),
    "longyearbyen_glacier_outlines_2010": os.path.join(
        _EXAMPLES_DIRECTORY, "Longyearbyen", "data", "glacier_mask", "CryoClim_GAO_SJ_2010.shp"
    ),
}

_FILEPATHS_PROCESSED = {
    "longyearbyen_ddem": os.path.join(_EXAMPLES_DIRECTORY, "Longyearbyen", "processed", "dDEM_2009_minus_1990.tif"),
    "longyearbyen_tba_dem_coreg": os.path.join(_EXAMPLES_DIRECTORY, "Longyearbyen", "processed", "DEM_1990_coreg.tif"),
}

available = list(_FILEPATHS_DATA.keys()) + list(_FILEPATHS_PROCESSED.keys())


def download_and_extract_tarball(dir: str, target_dir: str, overwrite: bool = False) -> None:
    """
    Helper function to download and extract a tarball from a given URL.

    :param dir: the directory to import.
    :param target_dir: The directory to extract the files into.
    :param overwrite: Whether to overwrite existing files.
    """

    # Exit code if files already exist
    if not overwrite and os.path.exists(target_dir) and os.listdir(target_dir):
        return

    if overwrite and os.path.exists(target_dir):
        # Clear existing files
        shutil.rmtree(target_dir)

    # Create a temporary directory to download the tarball
    temp_dir = tempfile.TemporaryDirectory()
    tar_path = os.path.join(temp_dir.name, "data.tar.gz")

    # Construct the URL with the commit hash
    url = f"{_DATA_REPO_URL}#commit={_COMMIT_HASH}"

    # Download the tarball
    response = urllib.request.urlopen(url)
    if response.getcode() == 200:
        with open(tar_path, "wb") as outfile:
            outfile.write(response.read())
    else:
        raise ValueError(f"Failed to download data: {response.status_code}")

    # Extract the tarball
    with tarfile.open(tar_path) as tar:
        tar.extractall(temp_dir.name)

    # Find the first directory inside the extracted tarball
    extracted_dir = os.path.join(
        temp_dir.name,
        [dirname for dirname in os.listdir(temp_dir.name) if os.path.isdir(os.path.join(temp_dir.name, dirname))][0],
        dir,
    )

    # Copy the extracted data to the target directory
    shutil.copytree(extracted_dir, target_dir)


def download_longyearbyen_examples(overwrite: bool = False) -> None:
    """
    Fetch the Longyearbyen example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    target_dir = os.path.join(_EXAMPLES_DIRECTORY, "Longyearbyen", "data")
    download_and_extract_tarball(dir="data/Longyearbyen", target_dir=target_dir, overwrite=overwrite)


def process_coregistered_examples(name: str, overwrite: bool = False) -> None:
    """
    Process the Longyearbyen example files into a dDEM (to avoid repeating this in many test/documentation steps).

    :param name: Name of test data
    :param overwrite: Do not download the files again if they already exist.
    """

    # If the file called already exists and overwrite is False, do nothing
    if not overwrite and os.path.isfile(_FILEPATHS_PROCESSED[name]):
        return

    # Check that data is downloaded before attempting processing
    download_longyearbyen_examples(overwrite=False)

    # If the ddem file does not exist, create it
    if not os.path.isfile(_FILEPATHS_PROCESSED["longyearbyen_ddem"]):
        reference_raster = gu.Raster(_FILEPATHS_DATA["longyearbyen_ref_dem"])
        to_be_aligned_raster = gu.Raster(_FILEPATHS_DATA["longyearbyen_tba_dem"])
        glacier_mask = gu.Vector(_FILEPATHS_DATA["longyearbyen_glacier_outlines"])
        inlier_mask = ~glacier_mask.create_mask(reference_raster)

        nuth_kaab = xdem.coreg.NuthKaab(offset_threshold=0.005)
        nuth_kaab.fit(reference_raster, to_be_aligned_raster, inlier_mask=inlier_mask, random_state=42)

        aligned_raster = nuth_kaab.apply(to_be_aligned_raster, resample=True)

        diff = reference_raster - aligned_raster

        # Save it so that future calls won't need to recreate the file
        os.makedirs(os.path.dirname(_FILEPATHS_PROCESSED["longyearbyen_ddem"]), exist_ok=True)
        diff.save(_FILEPATHS_PROCESSED["longyearbyen_ddem"])

    # If the tba_dem_coreg file does not exist, create it
    if not os.path.isfile(_FILEPATHS_PROCESSED["longyearbyen_tba_dem_coreg"]):

        dem_2009 = xdem.DEM(get_path("longyearbyen_ref_dem"), silent=True)
        ddem = xdem.DEM(get_path("longyearbyen_ddem"), silent=True)

        # Save it so that future calls won't need to recreate the file
        (dem_2009 - ddem).save(_FILEPATHS_PROCESSED["longyearbyen_tba_dem_coreg"])


def get_path(name: str) -> str:
    """
    Get path of example data. List of available files can be found in "examples.available".
    :param name: Name of test data
    :return:
    """

    if name in list(_FILEPATHS_DATA.keys()):
        download_longyearbyen_examples()
        return _FILEPATHS_DATA[name]
    elif name in list(_FILEPATHS_PROCESSED.keys()):
        process_coregistered_examples(name)
        return _FILEPATHS_PROCESSED[name]
    else:
        raise ValueError(
            'Data name should be one of "'
            + '" , "'.join(list(_FILEPATHS_DATA.keys()) + list(_FILEPATHS_PROCESSED.keys()))
            + '".'
        )

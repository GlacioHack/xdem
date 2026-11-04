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
import os.path as op
import shutil
import tarfile
import tempfile
import urllib.request
from importlib.resources import as_file, files

import geoutils as gu

import xdem

_DATA_REPO_URL = "https://github.com/GlacioHack/xdem-data/"
_COMMIT_HASH = "ca0e87271925d28928526bbce200162f002d6a93"

# This directory needs to be created within xdem/ so that it works for an installed package as well
# importlib.resources.files helps take care of the relative path, no matter if package is dev-local or installed
_EXAMPLES_DIRECTORY = files("xdem").joinpath("example_data")  # type: ignore

# Relative filepaths to the example files
_FILEPATHS_DATA = {
    "longyearbyen_ref_dem": op.join("Longyearbyen", "data", "DEM_2009_ref.tif"),
    "longyearbyen_tba_dem": op.join("Longyearbyen", "data", "DEM_1990.tif"),
    "longyearbyen_glacier_outlines": op.join("Longyearbyen", "data", "glacier_mask", "CryoClim_GAO_SJ_1990.shp"),
    "longyearbyen_glacier_outlines_2010": op.join("Longyearbyen", "data", "glacier_mask", "CryoClim_GAO_SJ_2010.shp"),
    "longyearbyen_epc": op.join("Longyearbyen", "data", "EPC_IS2.gpkg"),
    "giza_dem": op.join("Giza", "data", "DSM.tif"),
}

_FILEPATHS_PROCESSED = {
    "longyearbyen_ddem": op.join("Longyearbyen", "processed", "dDEM_2009_minus_1990.tif"),
    "longyearbyen_tba_dem_coreg": op.join("Longyearbyen", "processed", "DEM_1990_coreg.tif"),
}

_FILEPATHS_ALL = _FILEPATHS_DATA.copy()
_FILEPATHS_ALL.update(_FILEPATHS_PROCESSED)

available = list(_FILEPATHS_ALL.keys())
_FILEPATHS_TEST = {
    k: op.join(
        op.dirname(v),
        op.splitext(op.basename(v))[0] + "_test" + op.splitext(op.basename(v))[1],
    )
    for k, v in _FILEPATHS_ALL.items()
}
available_test = list(_FILEPATHS_TEST.keys())

# IF MODIFIED, NEED TO BE ADJUSTED IN XDEM-DATA TO PRODUCE GDAL OUTPUTS AS WELL
_TEST_ICROP_BOUNDS = (475, 600, 545, 654)


def _get_default_output_dir() -> str:
    """
    Return the output directory by default
    :return output directory path
    """
    with as_file(_EXAMPLES_DIRECTORY) as examples_directory:
        return str(examples_directory)


def _download_and_extract_tarball(dir: str, target_dir: str) -> None:
    """
    Helper function to download and extract a tarball from a given URL.

    :param dir: the directory to import.
    :param target_dir: The directory to extract the files into.
    """

    # Clear existing files
    if op.exists(target_dir):
        shutil.rmtree(target_dir)

    # Create a temporary directory to download the tarball
    temp_dir = tempfile.TemporaryDirectory()
    tar_path = op.join(temp_dir.name, "data.tar.gz")

    # Construct the URL with the commit hash
    url = f"{_DATA_REPO_URL}/archive/{_COMMIT_HASH}.tar.gz"

    # Download the tarball
    response = urllib.request.urlopen(url)
    if response.getcode() == 200:
        with open(tar_path, "wb") as outfile:
            outfile.write(response.read())
    else:
        raise ValueError(f"Failed to download data: {response.status_code}")

    # Extract the tarball
    with tarfile.open(tar_path) as tar:
        tar.extractall(temp_dir.name, filter="data")

    # Find the first directory inside the extracted tarball
    extracted_dir = op.join(
        temp_dir.name,
        [dirname for dirname in os.listdir(temp_dir.name) if op.isdir(op.join(temp_dir.name, dirname))][0],
        dir,
    )

    # Copy the extracted data to the target directory
    shutil.copytree(extracted_dir, target_dir)


def _download_data_examples(name: str, output_dir: str, overwrite: bool = False) -> None:
    """
    Fetch the data example files.

    :param name: Name of data.
    :param output_dir: Path of the directory to save the data.
    :param overwrite: Whether to overwrite the files if they already exist.
    """

    data = name.split("_")[0].capitalize()
    target_dir = op.join(output_dir, data, "data")
    file_exists = op.exists(op.join(output_dir, _FILEPATHS_DATA[name]))
    if overwrite or not file_exists:
        _download_and_extract_tarball(dir="data/" + data, target_dir=target_dir)


def _process_longyearbyen_coreg_examples(output_dir: str, overwrite: bool = False) -> None:  # TODO change the name
    """
    Process the Longyearbyen example files into a dDEM (to avoid repeating this in many test/documentation steps).

    :param name: Name of test data.
    :param output_dir: Path of the directory to save the data.
    :param overwrite: Whether to overwrite the files if they already exist.
    """

    # If the ddem file does not exist, create it
    ddem_path = op.join(output_dir, _FILEPATHS_PROCESSED["longyearbyen_ddem"])
    if overwrite or not op.isfile(ddem_path):
        # Get (and download) inputs data
        reference_raster = gu.Raster(get_path("longyearbyen_ref_dem", output_dir=output_dir))
        to_be_aligned_raster = gu.Raster(get_path("longyearbyen_tba_dem", output_dir=output_dir))
        glacier_mask = gu.Vector(get_path("longyearbyen_glacier_outlines", output_dir=output_dir))
        inlier_mask = ~glacier_mask.create_mask(reference_raster)

        nuth_kaab = xdem.coreg.NuthKaab(tolerance_translation=0.005)
        nuth_kaab.fit(reference_raster, to_be_aligned_raster, inlier_mask=inlier_mask, random_state=42)

        aligned_raster = nuth_kaab.apply(to_be_aligned_raster, resample=True)

        diff = reference_raster - aligned_raster

        # Save it so that future calls won't need to recreate the file
        os.makedirs(op.dirname(op.join(output_dir, _FILEPATHS_PROCESSED["longyearbyen_ddem"])), exist_ok=True)
        diff.to_file(op.join(output_dir, _FILEPATHS_PROCESSED["longyearbyen_ddem"]))

    # If the tba_dem_coreg file does not exist, create it
    if overwrite or not op.isfile(op.join(output_dir, _FILEPATHS_PROCESSED["longyearbyen_tba_dem_coreg"])):

        dem_2009 = xdem.DEM(get_path("longyearbyen_ref_dem", output_dir), silent=True)
        ddem = xdem.DEM(ddem_path, silent=True)  # absolute path to avoid recursivity

        # Save it so that future calls won't need to recreate the file
        (dem_2009 - ddem).to_file(op.join(output_dir, _FILEPATHS_PROCESSED["longyearbyen_tba_dem_coreg"]))


def get_path(name: str, output_dir: str | None = None, overwrite: bool = False) -> str:
    """
    Get path of example data. List of available files can be found in "examples.available".
    :param name: Name of test data.
    :param output_dir: Path of the directory to save the data.
    :param overwrite: Whether to overwrite the files if they already exist.
    :return:
    """

    if output_dir is None:
        output_dir = _get_default_output_dir()

    if name in list(_FILEPATHS_DATA.keys()):
        _download_data_examples(name, output_dir, overwrite)
    elif name in list(_FILEPATHS_PROCESSED.keys()):
        # For the moment, this part is specific to longyearbyen data
        _process_longyearbyen_coreg_examples(output_dir, overwrite)
    else:
        raise ValueError(
            'Data name should be one of "'
            + '" , "'.join(list(_FILEPATHS_DATA.keys()) + list(_FILEPATHS_PROCESSED.keys()))
            + '".'
        )
    return op.join(output_dir, _FILEPATHS_ALL[name])


def get_all_data(output_dir: str | None = None) -> str:

    if output_dir is None:
        output_dir = _get_default_output_dir()

    for k in _FILEPATHS_DATA.keys():
        _download_data_examples(k, output_dir)

    return output_dir


def _crop_lonyearbyen_test_examples(output_dir: str, overwrite: bool = False) -> None:
    """
    Crop the Longyearbyen examples to use for fast tests.

    :param output_dir: Path of the directory to save the data.
    :param overwrite: Whether to overwrite the files if they already exist.
    """
    for k in _FILEPATHS_TEST.keys():

        # Verify if data is longyearbyen and need to be computed
        if not k.lower().startswith("longyearbyen") or (
            op.exists(op.join(output_dir, _FILEPATHS_TEST[k])) and not overwrite
        ):
            continue

        # Get geometry to crop to
        ref_dem_cropped = xdem.DEM(op.join(output_dir, _FILEPATHS_ALL["longyearbyen_ref_dem"])).icrop(
            _TEST_ICROP_BOUNDS
        )

        # For rasters
        if os.path.basename(op.join(output_dir, _FILEPATHS_ALL[k])).split("_")[0] in ["DEM", "dDEM"]:
            cropped = gu.Raster(op.join(output_dir, _FILEPATHS_ALL[k])).crop(ref_dem_cropped.bounds)
        # For point cloud
        elif os.path.basename(op.join(output_dir, _FILEPATHS_ALL[k])).split("_")[0] == "EPC":
            pc = gu.PointCloud(op.join(output_dir, _FILEPATHS_ALL[k]), data_column="h_li")
            reprojected_dem_cropped = ref_dem_cropped.reproject(crs=pc.crs)
            cropped = pc.crop(reprojected_dem_cropped.bounds)
        # For vectors
        else:
            cropped = gu.Vector(op.join(output_dir, _FILEPATHS_ALL[k])).crop(ref_dem_cropped.bounds)

        cropped.to_file(op.join(output_dir, _FILEPATHS_TEST[k]))


def get_path_test(name: str, output_dir: str | None = None) -> str:
    """
    Get path of test data (reduced size). List of available files can be found in "examples.available".

    :param name: Name of test data.
    :param output_dir: Path of the directory to save the data.
    :return:
    """
    if output_dir is None:
        output_dir = _get_default_output_dir()

    if name in list(_FILEPATHS_TEST.keys()):
        # Download Longyearbyen raw data
        _download_data_examples(name="longyearbyen_ref_dem", output_dir=output_dir)

        # Process Longyearbyen data
        _process_longyearbyen_coreg_examples(output_dir=output_dir)

        # Crop them all
        _crop_lonyearbyen_test_examples(output_dir)

        return op.join(output_dir, _FILEPATHS_TEST[name])
    else:
        raise ValueError('Data name should be one of "' + '" , "'.join(list(_FILEPATHS_TEST.keys())) + '".')

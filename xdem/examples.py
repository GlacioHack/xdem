"""Utility functions to download and find example data."""
import os
import tarfile
import tempfile
import urllib.request
from distutils.dir_util import copy_tree

import geoutils as gu

import xdem

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


def download_longyearbyen_examples(overwrite: bool = False) -> None:
    """
    Fetch the Longyearbyen example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(_FILEPATHS_DATA.values()))):
        # print("Datasets exist")
        return

    # If we ask for overwrite, also remove the processed test data
    if overwrite:
        for fn in list(_FILEPATHS_PROCESSED.values()):
            if os.path.exists(fn):
                os.remove(fn)

    # Static commit hash to be bumped every time it needs to be.
    commit = "fd832bc2e366cf2ba8b543f7e43f90ee02384f4f"
    # The URL from which to download the repository
    url = f"https://github.com/GlacioHack/xdem-data/tarball/main#commit={commit}"

    # Create a temporary directory to extract the tarball in.
    temp_dir = tempfile.TemporaryDirectory()
    tar_path = os.path.join(temp_dir.name, "data.tar.gz")

    response = urllib.request.urlopen(url)
    # If the response was right, download the tarball to the temporary directory
    if response.getcode() == 200:
        with open(tar_path, "wb") as outfile:
            outfile.write(response.read())
    else:
        raise ValueError(f"Longyearbyen data fetch gave non-200 response: {response.status_code}")

    # Extract the tarball
    with tarfile.open(tar_path) as tar:
        tar.extractall(temp_dir.name)

    # Find the first directory in the temp_dir (should only be one) and construct the Longyearbyen data dir path.
    dir_name = os.path.join(
        temp_dir.name,
        [dirname for dirname in os.listdir(temp_dir.name) if os.path.isdir(os.path.join(temp_dir.name, dirname))][0],
        "data",
        "Longyearbyen",
    )

    # Copy the data to the examples directory.
    copy_tree(dir_name, os.path.join(_EXAMPLES_DIRECTORY, "Longyearbyen", "data"))


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

        nuth_kaab = xdem.coreg.NuthKaab()
        nuth_kaab.fit(reference_raster, to_be_aligned_raster, inlier_mask=inlier_mask)
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

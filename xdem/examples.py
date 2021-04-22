"""Utility functions to download and find example data."""
import os
import shutil
import tarfile
import tempfile
import urllib.request
from distutils.dir_util import copy_tree

EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", "examples/"))
# Absolute filepaths to the example files.
FILEPATHS = {
    "longyearbyen_ref_dem": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen/data/DEM_2009_ref.tif"),
    "longyearbyen_tba_dem": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen/data/DEM_1990.tif"),
    "longyearbyen_glacier_outlines": os.path.join(
        EXAMPLES_DIRECTORY,
        "Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp"
    ),
    "longyearbyen_glacier_outlines_2010": os.path.join(
        EXAMPLES_DIRECTORY,
        "Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_2010.shp"
    )
}


def download_longyearbyen_examples(overwrite: bool = False):
    """
    Fetch the Longyearbyen example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(FILEPATHS.values()))):
        print("Datasets exist")
        return

    # Static commit hash to be bumped every time it needs to be.
    commit = "321f84d5a67666f45a196a31a2697e22bfaf3c59"
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
        "data/Longyearbyen"
    )

    # Copy the data to the examples directory.
    copy_tree(dir_name, os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen/data"))

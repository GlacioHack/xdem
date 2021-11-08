"""Utility functions to download and find example data."""
import errno
import os
import shutil
import tarfile
import tempfile
import urllib.request
from distutils.dir_util import copy_tree

import numpy as np

import geoutils as gu
import xdem

EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
# Absolute filepaths to the example files.
FILEPATHS_DATA = {
    "longyearbyen_ref_dem": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen","data","DEM_2009_ref.tif"),
    "longyearbyen_tba_dem": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen","data","DEM_1990.tif"),
    "longyearbyen_glacier_outlines": os.path.join(
        EXAMPLES_DIRECTORY,
        "Longyearbyen","data","glacier_mask","CryoClim_GAO_SJ_1990.shp"
    ),
    "longyearbyen_glacier_outlines_2010": os.path.join(
        EXAMPLES_DIRECTORY,
        "Longyearbyen","data","glacier_mask","CryoClim_GAO_SJ_2010.shp"
    )}

FILEPATHS_PROCESSED = {
    "longyearbyen_ddem": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen","processed","dDEM_2009_minus_1990.tif"),
    "longyearbyen_tba_dem_coreg": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen", "processed", "DEM_1990_coreg.tif")
}


def download_longyearbyen_examples(overwrite: bool = False):
    """
    Fetch the Longyearbyen example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(FILEPATHS_DATA.values()))):
        # print("Datasets exist")
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
        "data","Longyearbyen"
    )

    # Copy the data to the examples directory.
    copy_tree(dir_name, os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen","data"))

def process_coregistered_examples(overwrite: bool =False):
    """
       Process the Longyearbyen example files into a dDEM (to avoid repeating this in many test/documentation steps).

       :param overwrite: Do not download the files again if they already exist.
       """

    if not overwrite and all(map(os.path.isfile, list(FILEPATHS_PROCESSED.values()))):
        # print("Processed data exists")
        return

    download_longyearbyen_examples(overwrite=False)

    # Run the coregistration if it hasn't been made yet.
    reference_raster = gu.georaster.Raster(FILEPATHS_DATA["longyearbyen_ref_dem"])
    to_be_aligned_raster = gu.georaster.Raster(FILEPATHS_DATA["longyearbyen_tba_dem"])
    glacier_mask = gu.geovector.Vector(FILEPATHS_DATA["longyearbyen_glacier_outlines"])
    inlier_mask = ~glacier_mask.create_mask(reference_raster)

    nuth_kaab = xdem.coreg.NuthKaab()
    nuth_kaab.fit(reference_raster.data, to_be_aligned_raster.data,
                  inlier_mask=inlier_mask, transform=reference_raster.transform)
    aligned_raster = nuth_kaab.apply(to_be_aligned_raster.data, transform=reference_raster.transform)
    aligned_raster.data[~np.isfinite(aligned_raster)] = reference_raster.nodata

    diff = reference_raster - \
           gu.Raster.from_array(aligned_raster.data, transform=reference_raster.transform, crs=reference_raster.crs, nodata=reference_raster.nodata)

    # Save it so that future calls won't need to recreate the file
    os.makedirs(os.path.dirname(FILEPATHS_PROCESSED['longyearbyen_ddem']), exist_ok=True)
    diff.save(FILEPATHS_PROCESSED['longyearbyen_ddem'])


def _get_longyearbyen_tba_coreg_dem() -> str:
    if not os.path.isfile(FILEPATHS_PROCESSED["longyearbyen_tba_dem_coreg"]):
        dem_2009 = xdem.DEM(get_path("longyearbyen_ref_dem"), silent=True)
        ddem = xdem.DEM(get_path("longyearbyen_ddem"), silent=True)

        (dem_2009 - ddem).save(FILEPATHS_PROCESSED["longyearbyen_tba_dem_coreg"])

    return FILEPATHS_PROCESSED["longyearbyen_tba_dem_coreg"]



def get_path(name: str) -> str:
    """
    Get path of example data
    :param name: Name of test data (listed in xdem/examples.py)
    :return:
    """
    if name == "longyearbyen_tba_dem_coreg":
        return _get_longyearbyen_tba_coreg_dem()

    if name in list(FILEPATHS_DATA.keys()):
        download_longyearbyen_examples()
        return FILEPATHS_DATA[name]
    elif name in list(FILEPATHS_PROCESSED.keys()):
        process_coregistered_examples()
        return FILEPATHS_PROCESSED[name]
    else:
        raise ValueError('Data name should be one of "'+'" , "'.join(list(FILEPATHS_DATA.keys())+list(FILEPATHS_PROCESSED.keys()))+'".')


import os
import tarfile
import tempfile
import urllib
from distutils.dir_util import copy_tree
from typing import Callable

import pytest

_TESTDATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "test_data"))

# Define a URL to the xdem-data repository's test data
_TESTDATA_REPO_URL = "https://github.com/vschaffn/xdem-data/tarball/2-richdem_gdal"
_COMMIT_HASH = "31a7159c982cec4b352f0de82bd4e0be61db3afe"


def download_test_data(overwrite: bool = False) -> None:
    """
    Download the entire test_data directory from the xdem-data repository.

    :param overwrite: If True, re-downloads the data even if it already exists.
    """
    if not overwrite and os.path.exists(_TESTDATA_DIRECTORY) and os.listdir(_TESTDATA_DIRECTORY):
        return  # Test data already exists

    # Clear the directory if overwrite is True
    if overwrite and os.path.exists(_TESTDATA_DIRECTORY):
        for root, dirs, files in os.walk(_TESTDATA_DIRECTORY, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    # Create a temporary directory to download the tarball
    temp_dir = tempfile.TemporaryDirectory()
    tar_path = os.path.join(temp_dir.name, "test_data.tar.gz")

    # Construct the URL with the commit hash
    url = f"{_TESTDATA_REPO_URL}#commit={_COMMIT_HASH}"

    response = urllib.request.urlopen(url)
    if response.getcode() == 200:
        with open(tar_path, "wb") as outfile:
            outfile.write(response.read())
    else:
        raise ValueError(f"Failed to download test data: {response.status_code}")

    # Extract the tarball
    with tarfile.open(tar_path) as tar:
        tar.extractall(temp_dir.name)

    # Copy the test_data directory to the target directory
    extracted_dir = os.path.join(
        temp_dir.name,
        [dirname for dirname in os.listdir(temp_dir.name) if os.path.isdir(os.path.join(temp_dir.name, dirname))][0],
        "test_data",
    )

    copy_tree(extracted_dir, _TESTDATA_DIRECTORY)


@pytest.fixture(scope="session")  # type: ignore
def get_test_data_path() -> Callable[[str], str]:
    def _get_test_data_path(filename: str, overwrite: bool = False) -> str:
        """Get file from test_data"""
        download_test_data(overwrite=overwrite)  # Ensure the test data is downloaded
        file_path = os.path.join(_TESTDATA_DIRECTORY, filename)

        if not os.path.exists(file_path):
            if overwrite:
                raise FileNotFoundError(f"The file {filename} was not found in the test_data directory.")
            file_path = _get_test_data_path(filename, overwrite=True)

        return file_path

    return _get_test_data_path

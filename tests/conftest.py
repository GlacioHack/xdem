import os
from typing import Callable

import pytest

from xdem.examples import download_and_extract_tarball

_TESTDATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "test_data"))


@pytest.fixture(scope="session")  # type: ignore
def get_test_data_path() -> Callable[[str], str]:
    def _get_test_data_path(filename: str, overwrite: bool = False) -> str:
        """Get file from test_data"""
        download_and_extract_tarball(dir="test_data", target_dir=_TESTDATA_DIRECTORY, overwrite=overwrite)
        file_path = os.path.join(_TESTDATA_DIRECTORY, filename)

        if not os.path.exists(file_path):
            if overwrite:
                raise FileNotFoundError(f"The file {filename} was not found in the test_data directory.")
            file_path = _get_test_data_path(filename, overwrite=True)

        return file_path

    return _get_test_data_path

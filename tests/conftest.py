import os
from typing import Callable

import numpy as np
import pytest

from xdem.examples import _download_and_extract_tarball
from xdem.terrain import get_terrain_attribute

_TESTDATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "test_data"))
_TESTOUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "test_output"))


@pytest.fixture(scope="session")  # type: ignore
def test_output_dir() -> str:

    os.makedirs(_TESTOUTPUT_DIRECTORY, exist_ok=True)

    """Return the path to the test output directory."""
    return _TESTOUTPUT_DIRECTORY


@pytest.fixture(scope="session")  # type: ignore
def get_test_data_path() -> Callable[[str], str]:
    def _get_test_data_path(filename: str, overwrite: bool = False) -> str:
        """Get file from test_data"""
        _download_and_extract_tarball(dir="test_data", target_dir=_TESTDATA_DIRECTORY, overwrite=overwrite)
        file_path = os.path.join(_TESTDATA_DIRECTORY, filename)

        if not os.path.exists(file_path):
            if overwrite:
                raise FileNotFoundError(f"The file {filename} was not found in the test_data directory.")
            file_path = _get_test_data_path(filename, overwrite=True)

        return file_path

    return _get_test_data_path


@pytest.fixture(scope="session", autouse=True)
def precompile_numba_functions():
    """Pre-compile Numba functions ahead of test execution to avoid multiple compilations."""

    # Define arbitrary DEM
    rng = np.random.default_rng(42)
    dem = rng.normal(size=(5, 5))

    # Trigger compile of surface fit attributes
    get_terrain_attribute(dem, resolution=1, attribute="slope", engine="numba")
    # Trigger compile of surface fit attributes
    get_terrain_attribute(dem, resolution=1, attribute="roughness", engine="numba")

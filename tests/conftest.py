import logging
import os
from typing import Any, Callable

import pytest

from xdem.examples import _download_and_extract_tarball

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


class LoggingWarningCollector(logging.Handler):
    """Helper class to collect logging warnings."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record: Any) -> None:
        self.records.append(record)


@pytest.fixture(autouse=True)  # type: ignore
def fail_on_logging_warnings(request: Any) -> Any:
    """Fixture used automatically in all tests to fail when a logging exceptions of WARNING or above is raised."""

    # The collector is required to avoid teardown, hookwrapper or plugin issues (we collect and fail later)
    collector = LoggingWarningCollector()
    root = logging.getLogger()
    root.addHandler(collector)

    # Run test
    yield

    root.removeHandler(collector)

    # Allow opt-out
    if request.node.get_closest_marker("allow_logging_warnings"):
        return

    # Categorize bad tests
    # IGNORED = ("rasterio",)   # If we want to add a list of "IGNORED" packages in the future
    bad = [
        r
        for r in collector.records
        if r.levelno >= logging.WARNING
        # and not r.name.startswith(IGNORED)
    ]

    # Fail on those exceptions and report exception level, name and message
    if bad:
        msgs = "\n".join(f"{r.levelname}:{r.name}:{r.getMessage()}" for r in bad)
        pytest.fail(
            "Logging warning/error detected:\n" + msgs,
            pytrace=False,
        )

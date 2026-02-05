import logging
import os
import re
from typing import Any, Callable, Pattern, Union

import pytest

from xdem.examples import _download_and_extract_tarball

_TESTDATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "test_data"))
_TESTOUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "test_output"))


@pytest.fixture(scope="session")
def test_output_dir() -> str:

    os.makedirs(_TESTOUTPUT_DIRECTORY, exist_ok=True)

    """Return the path to the test output directory."""
    return _TESTOUTPUT_DIRECTORY


@pytest.fixture(scope="session")
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


@pytest.fixture(autouse=True)
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
        and not getattr(r, "expected", False)  # Skip expected logging warnings (tagged manually in the tests)
        # and not r.name.startswith(IGNORED)
    ]

    # Fail on those exceptions and report exception level, name and message
    if bad:
        msgs = "\n".join(f"{r.levelname}:{r.name}:{r.getMessage()}" for r in bad)
        pytest.fail(
            "Logging warning/error detected:\n" + msgs,
            pytrace=False,
        )


def _assert_and_allow_log(
    caplog: Any,
    *,
    level: int = logging.WARNING,
    match: Union[str, Pattern[str]],
    logger: str | None = None,
) -> None:
    """Helper function to capture and check logging exceptions, avoiding failures from the global collector above."""

    # Compile regex match
    if isinstance(match, str):
        match = re.compile(match)

    # Find matches
    matches = [
        r
        for r in caplog.records
        if r.levelno == level and match.search(r.getMessage()) and (logger is None or r.name == logger)
    ]

    # Assert matches, otherwise return a helpful message for debugging
    assert matches, (
        f"Expected log not found.\n"
        f"  level: {logging.getLevelName(level)}\n"
        f"  logger: {logger or '*'}\n"
        f"  pattern: {match.pattern}\n"
        f"Logs seen:\n" + "\n".join(f"{r.levelname}:{r.name}:{r.getMessage()}" for r in caplog.records)
    )

    # If assertion passed, set warnings as expected for the global collector (function above)
    for r in matches:
        r.expected = True


@pytest.fixture
def assert_and_allow_log() -> Any:
    return _assert_and_allow_log

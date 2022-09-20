"""Test the xdem.misc functions."""
from __future__ import annotations

import warnings

import pytest

import xdem
import xdem.misc


@pytest.mark.parametrize("deprecation_increment", [-1, 0, 1, None])  # type: ignore
@pytest.mark.parametrize("details", [None, "It was completely useless!", "dunnowhy"])  # type: ignore
def test_deprecate(deprecation_increment: int | None, details: str | None) -> None:
    """
    Test the deprecation warnings/errors.

    If the removal_version is larger than the current, it should warn.
    If the removal_version is smaller or equal, it should raise an error.

    :param deprecation_increment: The version number relative to the current version.
    :param details: An optional explanation for the description.
    """
    warnings.simplefilter("error")

    current_version = xdem.version.version

    # Set the removal version to be the current version plus the increment (e.g. 0.0.5 + 1 -> 0.0.6)
    removal_version = (
        current_version[:-1] + str(int(current_version.rsplit(".", 1)[1]) + deprecation_increment)
        if deprecation_increment is not None
        else None
    )

    # Define a function with no use that is marked as deprecated.
    @xdem.misc.deprecate(removal_version, details=details)  # type: ignore
    def useless_func() -> int:
        return 1

    # If True, a warning is expected. If False, a ValueError is expected.
    should_warn = removal_version is None or removal_version > current_version

    # Add the expected text depending on the parametrization.
    text = (
        "Call to deprecated function 'useless_func'."
        if should_warn
        else f"Deprecated function 'useless_func' was removed in {removal_version}."
    )

    if details is not None:
        text += " " + details.strip().capitalize()

        if not any(text.endswith(c) for c in ".!?"):
            text += "."

    if should_warn and removal_version is not None:
        text += f" This functionality will be removed in version {removal_version}."
    elif not should_warn:
        text += f" Current version: {current_version}."

    # Expect either a warning or an exception with the right text.
    if should_warn:
        with pytest.warns(DeprecationWarning, match="^" + text + "$"):
            useless_func()
    else:
        with pytest.raises(ValueError, match="^" + text + "$"):
            useless_func()

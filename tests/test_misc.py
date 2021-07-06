"""Test the xdem.misc functions."""
from __future__ import annotations

import warnings

import pytest

import xdem
import xdem.misc


@pytest.mark.parametrize("deprecation_increment", [-1, 0, 1, None])
@pytest.mark.parametrize("details", [None, "It was completely useless!", "dunnowhy"])
def test_deprecate(deprecation_increment: int | None, details: str | None) -> None:
    """
    Test the deprecation warnings/errors.

    If the version is larger than the current, it should warn.
    If the version is smaller or equal, it should raise an error.

    :param deprecation_increment: The version number relative to the current version.
    """
    warnings.simplefilter("error")

    current_version = xdem.version.version

    # Set the deprecation_version to be the current version plus the increment (e.g. 0.0.5 + 1 -> 0.0.6)
    deprecation_version = (
        current_version[:-1] + str(int(current_version.rsplit(".", 1)[1]) + deprecation_increment)
        if deprecation_increment is not None
        else None
    )

    @xdem.misc.deprecate(deprecation_version, details=details)
    def useless_func() -> int:
        return 1

    should_warn = deprecation_version is None or deprecation_version > current_version

    text = (
        "Call to deprecated function 'useless_func'."
        if should_warn
        else f"Function 'useless_func' was deprecated in {deprecation_version}."
    )

    if details is not None:
        text += " " + details.strip().capitalize()

        if not any(text.endswith(c) for c in ".!?"):
            text += "."

    
    if should_warn and deprecation_version is not None:
        text += f" This functionality will be removed in version {deprecation_version}."
    elif not should_warn:
        text += f" Current version: {current_version}."


    if should_warn:
        with pytest.warns(DeprecationWarning, match=text):
            useless_func()
    else:
        with pytest.raises(ValueError, match=text):
            useless_func()

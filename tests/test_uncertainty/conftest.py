"""Shared fixtures for uncertainty tests."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

import xdem

# List all coreg methods for tests once
_COREG_METHODS = [
    xdem.coreg.VerticalShift,
    xdem.coreg.NuthKaab,
    xdem.coreg.LZD,
    xdem.coreg.ICP,
    xdem.coreg.CPD,
    xdem.coreg.DhMinimize,
]


def _make_coreg_method(method: type[xdem.coreg.AffineCoreg]) -> xdem.coreg.AffineCoreg:
    """Create a coregistration method with full sample, and check the warning for its deprecated alias."""

    if method is xdem.coreg.DhMinimize:
        with pytest.warns(DeprecationWarning, match="DhMinimize is deprecated"):
            return method(subsample=1)
    return method(subsample=1)


@pytest.fixture(params=_COREG_METHODS, ids=lambda method: method.__name__)
def coreg_method(request: pytest.FixtureRequest) -> xdem.coreg.AffineCoreg:
    """Create each affine method using all observations."""

    return _make_coreg_method(request.param)


# Compare every solver through the function and check forwarding with LZD
_PRECOREG_CASES = [
    pytest.param((method, "function", True), id=f"{method.__name__}-function-enabled") for method in _COREG_METHODS
]
_PRECOREG_CASES.extend(
    [
        pytest.param((xdem.coreg.LZD, "function", False), id="function-disabled"),
        pytest.param((xdem.coreg.LZD, "method", True), id="method-enabled"),
        pytest.param((xdem.coreg.LZD, "method", False), id="method-disabled"),
        pytest.param((xdem.coreg.LZD, "accessor", True), id="accessor-enabled"),
        pytest.param((xdem.coreg.LZD, "accessor", False), id="accessor-disabled"),
    ]
)


@pytest.fixture(params=_PRECOREG_CASES)
def precoreg_case(request: pytest.FixtureRequest) -> tuple[xdem.coreg.AffineCoreg, str, bool]:
    """Select a solver comparison or a check of prealignment options through an entrypoint."""

    method, entrypoint, enabled = request.param
    return _make_coreg_method(method), entrypoint, enabled


@pytest.fixture
def curved_dem() -> xdem.DEM:
    """Create a curved DEM with observable translations and rotations for uncertainty workflows."""

    rows, columns = np.mgrid[:18, :20]
    values = 5 * np.sin(columns / 4) + 3 * np.cos(rows / 3) + 2 * np.sin((columns + rows) / 7)
    transform = Affine(2, 0, 500_000, 0, -2, 7_000_000)
    return xdem.DEM.from_array(values, transform, 32632, nodata=-9999)

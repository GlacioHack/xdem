"""Tests that xDEM just links to GeoUtils error structure."""

from __future__ import annotations

import geoutils as gu

import xdem


class TestErrorStructure:
    """Test module to check link to GeoUtils error structure."""

    def test_shared_classes__identity(self) -> None:
        """Checks that xDEM exports the GeoUtils error classes directly."""

        assert xdem.ErrorStructure is gu.ErrorStructure
        assert xdem.ErrorComponent is gu.ErrorComponent
        assert xdem.ErrorMagnitude is gu.ErrorMagnitude

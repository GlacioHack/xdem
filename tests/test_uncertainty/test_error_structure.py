"""Tests for GeoUtils error models used by xDEM elevation recipes."""

from __future__ import annotations

import geoutils as gu

import xdem


class TestErrorStructureRepresentations:
    """Test module for source model classes shared with GeoUtils."""

    def test_shared_classes__same_objects_in_both_packages(self) -> None:
        """Checks that xDEM exports the GeoUtils source model classes directly."""

        # The same initial error model can enter either package's propagation call
        assert xdem.ErrorStructure is gu.ErrorStructure
        assert xdem.ErrorComponent is gu.ErrorComponent
        assert xdem.ErrorMagnitude is gu.ErrorMagnitude

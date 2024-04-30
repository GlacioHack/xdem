from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
import rasterio as rio

from xdem.coreg.base import (
    _get_inlier_mask,
    _get_valid_data_mask,
    _select_transform_crs,
    _validate_masks,
)


def test__get_valid_data_mask_returns_correct_mask() -> None:
    """Test that get_valid_data_mask returns a mask for invalid and nodata values."""
    no_data = -999
    data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, no_data, 8],
            [9, np.nan, 11, 12],
            [13, 14, 15, np.nan],
        ]
    )
    da_data = da.from_array(data, chunks=(2, 2))

    expected_mask = np.array(
        [
            [False, False, False, False],
            [False, False, True, False],
            [False, True, False, False],
            [False, False, False, True],
        ]
    )

    # no_data exchanged for np.nan
    expected_maked_data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, np.nan, 8],
            [9, np.nan, 11, 12],
            [13, 14, 15, np.nan],
        ]
    )

    mask, masked_data = _get_valid_data_mask(data=da_data, nodata_value=no_data)

    assert np.all(mask.compute() == expected_mask)
    assert np.allclose(masked_data.compute(), expected_maked_data, equal_nan=True)


def test__get_valid_data_mask_raises_on_all_invalid_data() -> None:
    """Test that get_valid_data_mask raises for all invalid data."""
    no_data = -999
    data = np.full(shape=(4, 4), fill_value=np.nan)
    da_data = da.from_array(data, chunks=(2, 2))

    with pytest.raises(expected_exception=ValueError):
        _ = _get_valid_data_mask(data=da_data, nodata_value=no_data)  # .compute()


@pytest.mark.skip(reason="pytest fails test if a UserWarning is issued.")  # type: ignore[misc]
def test__select_transform_crs_selects_correct_crs() -> None:
    """Test _select_transform_crs selects the correct crs."""

    # TODO fix this test

    # not using pytest.mark.parametrize because of compatibility issues with pytest > 8.0.0
    # did not want to pin the pytest version for dependencies because of this.
    # https://github.com/smarie/python-pytest-cases/issues/330

    # the priority should be: crs_ref, crs_other, crs
    # pairs of crs_ref, crs_other, crs, expected
    epsg_pairs = [
        (3246, 4326, 3005, 3246),
        (None, 4326, 3005, 4326),
        (None, None, 3005, 3005),
    ]
    mock_transform = Mock(rio.transform.Affine)  # we dont care about the transform in this test

    for epsg_ref, epsg_other, epsg, expected in epsg_pairs:
        _, crs = _select_transform_crs(
            transform=mock_transform,
            crs=rio.crs.CRS.from_epsg(epsg),
            transform_reference=mock_transform,
            transform_other=mock_transform,
            crs_reference=rio.crs.CRS.from_epsg(epsg_ref),
            crs_other=rio.crs.CRS.from_epsg(epsg_other),
        )
        assert crs.to_epsg() == expected


def test__select_transform_crs_selects_correct_transform() -> None:
    """Test _select_transform_crs selects the correct transform."""
    # TODO
    pass


def test__get_inlier_mask_with_correct_mask() -> None:
    """Test _get_inlier_mask returns mask if it is the correct type and has inliers."""

    mask = np.full((4, 4), fill_value=True, dtype=bool)
    da_mask = da.from_array(mask, chunks=(2, 2))
    returned_mask = _get_inlier_mask(mask=da_mask).compute()
    assert np.all(np.equal(mask, returned_mask))


def test__get_inlier_mask_raises_witout_inliers() -> None:
    """Test _get_inlier_mask raises without inliers."""

    mask = np.full((4, 4), fill_value=False, dtype=bool)
    da_mask = da.from_array(mask, chunks=(2, 2))
    with pytest.raises(expected_exception=ValueError):
        _ = _get_inlier_mask(mask=da_mask)


def test__get_inlier_mask_raises_with_wrong_mask_type() -> None:
    """Test _get_inlier_mask raises with wrong mask type."""

    mask = np.full((4, 4), fill_value=1, dtype=np.uint32)
    da_mask = da.from_array(mask, chunks=(2, 2))
    with pytest.raises(expected_exception=TypeError):
        _ = _get_inlier_mask(mask=da_mask)


def test__validate_masks_does_not_raise_for_valid_data() -> None:
    """Test _validate_masks does not raise with valid data."""

    ref_mask = np.full((4, 4), False, dtype=bool)
    ref_mask[:2] = True
    da_ref_mask = da.from_array(ref_mask, chunks=(2, 2))

    tba_mask = np.full((4, 4), False, dtype=bool)
    da_tba_mask = da.from_array(tba_mask, chunks=(2, 2))

    inlier_mask = np.full((4, 4), True, dtype=bool)
    da_inlier_mask = da.from_array(inlier_mask, chunks=(2, 2))

    _validate_masks(inlier_mask=da_inlier_mask, ref_mask=da_ref_mask, tba_mask=da_tba_mask)


def test__validate_masks_raises_with_combined_all_invalid_data() -> None:
    """Test _validate_masks raises with combined all invalid data."""

    ref_mask = np.full((4, 4), False, dtype=bool)
    ref_mask[:2] = True
    da_ref_mask = da.from_array(ref_mask, chunks=(2, 2))

    tba_mask = np.full((4, 4), False, dtype=bool)
    tba_mask[2:] = True
    da_tba_mask = da.from_array(tba_mask, chunks=(2, 2))

    inlier_mask = np.full((4, 4), True, dtype=bool)
    da_inlier_mask = da.from_array(inlier_mask, chunks=(2, 2))

    with pytest.raises(ValueError):
        _validate_masks(inlier_mask=da_inlier_mask, ref_mask=da_ref_mask, tba_mask=da_tba_mask)


def test__validate_masks_does_raises_with_combined_invalid_data_from_inlier_mask() -> None:
    """Test _validate_masks does not raise with valid data."""

    ref_mask = np.full((4, 4), False, dtype=bool)
    ref_mask[:2] = True
    da_ref_mask = da.from_array(ref_mask, chunks=(2, 2))

    tba_mask = np.full((4, 4), False, dtype=bool)
    da_tba_mask = da.from_array(tba_mask, chunks=(2, 2))

    inlier_mask = np.full((4, 4), True, dtype=bool)
    inlier_mask[2:] = False

    da_inlier_mask = da.from_array(inlier_mask, chunks=(2, 2))
    with pytest.raises(ValueError):
        _validate_masks(inlier_mask=da_inlier_mask, ref_mask=da_ref_mask, tba_mask=da_tba_mask)

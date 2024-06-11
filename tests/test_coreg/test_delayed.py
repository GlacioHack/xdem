from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
import rasterio as rio

from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import (
    _select_transform_crs,
    get_valid_data,
    mask_data,
    valid_data_darr,
)


@pytest.mark.filterwarnings("ignore::UserWarning")  # type: ignore [misc]
@pytest.mark.parametrize(  # type: ignore [misc]
    "epsg_ref, epsg_other, epsg, expected",
    [
        (3246, 4326, 3005, 3246),
        (None, 4326, 3005, 4326),
        (None, None, 3005, 3005),
    ],
)
def test__select_transform_crs_selects_correct_crs(
    epsg_ref: int | None, epsg_other: int | None, epsg: int, expected: int
) -> None:
    """Test _select_transform_crs selects the correct crs."""
    mock_transform = Mock(rio.transform.Affine)  # we dont care about the transform in this test

    # for epsg_ref, epsg_other, epsg, expected in epsg_pairs:
    _, crs = _select_transform_crs(
        transform=mock_transform,
        crs=rio.crs.CRS.from_epsg(epsg),
        transform_reference=mock_transform,
        transform_other=mock_transform,
        crs_reference=rio.crs.CRS.from_epsg(epsg_ref) if epsg_ref is not None else epsg_ref,
        crs_other=rio.crs.CRS.from_epsg(epsg_other) if epsg_other is not None else epsg_other,
    )
    assert crs.to_epsg() == expected


def test__select_transform_crs_selects_correct_transform() -> None:
    """Test _select_transform_crs selects the correct transform."""
    # TODO
    pass


@pytest.mark.parametrize(  # type: ignore[misc]
    "input,nodata,expected",
    [
        (
            np.array([np.nan, 1, -100, 1]),
            -100,
            np.array([np.nan, 1, np.nan, 1]),
        ),
        (
            np.array([1, 1, -100, 1]),
            -100,
            np.array([1, 1, np.nan, 1]),
        ),
        (
            np.array([np.nan, 1, 1, 1]),
            -100,
            np.array([np.nan, 1, 1, 1]),
        ),
    ],
)
def test_mask_data(input: NDArrayf, nodata: int, expected: NDArrayf) -> None:
    """Test that mask_data masks the correct values."""
    output = mask_data(data=input, nodata=nodata)
    assert np.array_equal(output, expected, equal_nan=True)


@pytest.mark.parametrize(  # type: ignore [misc]
    "input_arrays,nodatas,expected",
    [
        (
            (np.array([np.nan, 1, -100, 1]),),
            (-100,),
            np.array([False, True, False, True]),
        ),
        (
            (
                np.array([np.nan, 1, -100, 1]),
                np.array([1, -200, 1, 1]),
            ),
            (-100, -200),
            np.array([False, False, False, True]),
        ),
        (
            (
                np.array([np.nan, 1, -100, 1]),
                np.array([1, -200, 1, 1]),
                np.array([1, 1, 1, -400]),
            ),
            (-100, -200, -400),
            np.array([False, False, False, False]),
        ),
    ],
)
def test_get_valid_data(input_arrays: tuple[NDArrayf], nodatas: tuple[int], expected: NDArrayb) -> None:
    """Test get_valid_data returns correct output."""
    output = get_valid_data(*input_arrays, nodatas=nodatas)
    assert np.array_equal(output, expected, equal_nan=True)


@pytest.mark.parametrize(  # type: ignore [misc]
    "input_arrays,mask,nodatas,expected",
    [
        (
            (
                da.from_array(np.array([1, 1, -100, 1]), chunks=2),
                da.from_array(np.array([1, 1, -200, 1]), chunks=2),
            ),
            None,
            (-100, -200),
            np.array([True, True, False, True]),
        ),
        (
            (
                da.from_array(np.array([1, 1, -100, 1]), chunks=2),
                da.from_array(np.array([1, 1, -200, 1]), chunks=2),
            ),
            da.from_array([False, True, True, True]),
            (-100, -200),
            np.array([False, True, False, True]),
        ),
    ],
)
def test_valid_data_darr(
    input_arrays: tuple[NDArrayf], mask: NDArrayb, nodatas: tuple[int], expected: NDArrayb
) -> None:
    """Test valid_data_darr returns correct output."""
    output = valid_data_darr(*input_arrays, mask=mask, nodatas=nodatas).compute()
    assert np.array_equal(output, expected, equal_nan=True)

"""Test module for the DEMBase class."""
from __future__ import annotations

import warnings
from typing import Any

import pytest
import numpy as np
import xarray as xr
from geoutils import Vector

from xdem import DEM, open_dem
from xdem import examples

class TestDEMBase:

    pass


def equal_xr_dem(ds: xr.DataArray, rast: DEM, warn_failure_reason: bool = True) -> bool:
    """Check equality of a Raster object and Xarray object"""

    # TODO: Move to raster_equal?
    equalities = [
        np.allclose(ds.data, rast.get_nanarray(), equal_nan=True),
        ds.rst.transform == rast.transform,
        ds.rst.crs == rast.crs,
        ds.rst.nodata == rast.nodata,
    ]

    names = ["data", "transform", "crs", "nodata"]

    complete_equality = all(equalities)

    if not complete_equality and warn_failure_reason:
        where_fail = np.nonzero(~np.array(equalities))[0]
        warnings.warn(
            category=UserWarning, message=f"Equality failed for: {', '.join([names[w] for w in where_fail])}."
        )
        print(f"Equality failed for: {', '.join([names[w] for w in where_fail])}.")

    print(np.count_nonzero(np.isfinite(ds.data) != np.isfinite(rast.get_nanarray())))
    print(np.nanmin(ds.data - rast.get_nanarray()))
    print(ds.data)

    return complete_equality

def output_equal(output1: Any, output2: Any) -> bool:
    """Return equality of different output types."""

    # For two vectors
    if isinstance(output1, Vector) and isinstance(output2, Vector):
        return output1.vector_equal(output2)

    # For two raster: Xarray or Raster objects
    elif isinstance(output1, DEM) and isinstance(output2, DEM):
        return output1.raster_equal(output2)
    elif isinstance(output1, DEM) and isinstance(output2, xr.DataArray):
        return equal_xr_dem(ds=output2, rast=output1)
    elif isinstance(output1, xr.DataArray) and isinstance(output2, DEM):
        return equal_xr_dem(ds=output1, rast=output2)

    # For arrays
    elif isinstance(output1, np.ndarray):
        return np.array_equal(output1, output2, equal_nan=True)

    # For tuple of arrays
    elif isinstance(output1, tuple) and isinstance(output1[0], np.ndarray):
        return np.array_equal(np.array(output1), np.array(output2), equal_nan=True)

    # For any other object type
    else:
        return output1 == output2

class TestClassVsAccessorConsistency:
    """
    Test class to check the consistency between the outputs of the DEM class and Xarray accessor for the same
    attributes or methods.

    All shared attributes should be the same.
    All operations manipulating the array should yield a comparable results, accounting for the fact that Raster class
    relies on masked-arrays and the Xarray accessor on NaN arrays.
    """

    # Run tests for different rasters
    longyearbyen_path = examples.get_path("longyearbyen_ref_dem")

    # Test common attributes
    attributes_raster = ["crs", "transform", "nodata", "area_or_point", "res", "count", "height", "width", "footprint",
                         "shape", "bands", "indexes", "_is_xr", "is_loaded"]
    attributes_dem = ["vcrs", "vcrs_grid", "vcrs_name"]
    attributes = attributes_dem + attributes_raster

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("attr", attributes)  # type: ignore
    def test_attributes(self, path_dem: str, attr: str) -> None:
        """Test that attributes of the two objects are exactly the same."""

        # Open
        ds = open_dem(path_dem)
        dem = DEM(path_dem)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get attribute for each object
        output_dem = getattr(dem, attr)
        output_ds = getattr(getattr(ds, "dem"), attr)

        # Assert equality
        if attr != "_is_xr":  # Only attribute that is (purposely) not the same, but the opposite
            assert output_equal(output_dem, output_ds)
        else:
            assert output_dem != output_ds


    # Test common methods
    methods_and_args = {
        "slope": {},
        "aspect": {},
        "hillshade": {},
    }

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("method", list(methods_and_args.keys()))  # type: ignore
    def test_methods(self, path_dem: str, method: str) -> None:
        """
        Test that the outputs of the two objects are exactly the same
        (converted for the case of a DEM/vector output, as it can be a Xarray/GeoPandas object or DEM/Vector).
        """

        # Open both objects
        ds = open_dem(path_dem)
        dem = DEM(path_dem)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        args = self.methods_and_args[method].copy()

        # Apply method for each class
        output_dem = getattr(dem, method)(**args)
        output_ds = getattr(getattr(ds, "dem"), method)(**args)

        # Assert equality of output
        assert output_equal(output_dem, output_ds)

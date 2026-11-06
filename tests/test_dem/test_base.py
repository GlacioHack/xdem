"""Test module for the DEMBase class."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from geoutils import Raster, Vector
from geoutils.raster import MultiprocConfig
from pandas.testing import assert_frame_equal
from pyproj import CRS

from xdem import DEM, EPC, coreg, examples, open_dem
from xdem.dem.base import DEMBase
from xdem.dem.xr_accessor import DEMAccessor
from xdem.vcrs import _VerticalReference


def assert_output_equal(output1: Any, output2: Any, use_allclose: bool = False, strict_masked: bool = True) -> None:
    """Return equality of different output types."""

    # For two vectors
    if isinstance(output1, Vector) and isinstance(output2, Vector):
        assert output1.vector_equal(output2)

    # For two raster-like outputs: Xarray or DEM objects
    elif isinstance(output1, (Raster, xr.DataArray)):
        if use_allclose:
            assert output1.raster_allclose(output2, warn_failure_reason=True, strict_masked=strict_masked)
        else:
            assert output1.raster_equal(output2, warn_failure_reason=True, strict_masked=strict_masked)

    # For arrays
    elif isinstance(output1, np.ndarray):
        if np.ma.isMaskedArray(output1):
            output1 = output1.filled(np.nan)
        if np.ma.isMaskedArray(output2):
            output2 = output2.filled(np.nan)
        if use_allclose:
            assert np.allclose(output1, output2, equal_nan=True)
        else:
            assert np.array_equal(output1, output2, equal_nan=True)

    # For tuple of arrays
    elif isinstance(output1, tuple) and len(output1) > 0 and isinstance(output1[0], np.ndarray):
        assert np.array_equal(np.array(output1), np.array(output2), equal_nan=True)

    # For a list of raster-like outputs
    elif isinstance(output1, list) and len(output1) > 0 and isinstance(output1[0], (Raster, xr.DataArray)):
        assert len(output1) == len(output2)
        for out1, out2 in zip(output1, output2):
            assert_output_equal(out1, out2, use_allclose=use_allclose, strict_masked=strict_masked)

    # For a dictionary of numeric values
    elif isinstance(output1, dict):
        df1 = pd.DataFrame(index=[0], data=output1)
        df2 = pd.DataFrame(index=[0], data=output2)
        assert_frame_equal(df1, df2, check_dtype=False)

    # For any other object type
    else:
        assert output1 == output2


def should_be_loaded(method: str, args: dict[str, Any], noload: list[str], noload_allowed_args: dict[str, Any]) -> bool:
    """Helper function to check without a method input/output should be loaded or not, based on input dictionaries."""

    # For method where the behaviour is independent of their arguments
    if method not in noload_allowed_args:
        # If the method has a single behaviour, simply check if it belongs in the list
        should_output_be_loaded = method not in noload
    # For method where the behaviour depends on their arguments
    else:
        # Get relevant method arguments
        allowed = noload_allowed_args[method]
        # If any value is different from the list of values in the allowed dictionary, it should load
        any_different = not all(
            (not isinstance(args[k], np.ndarray) and args[k] in allowed[k]) for k in allowed if k in args
        )
        should_output_be_loaded = any_different

    return should_output_be_loaded


class NeedsTestError(ValueError):
    """Error to remember to add a test when a shared DEM elevation method is added."""


class TestDEMInheritance:
    """
    Test that DEM and its Xarray accessor use the shared DEM base classes.

    This class tests:
    - ``DEM`` and ``DEMAccessor`` inheritance from ``DEMBase`` and ``_VerticalReference``,
    - The absence of unused overrides for methods already implemented by those base classes.

    The consistency classes below test the behavior of the shared methods.
    """

    def test_shared_method_ownership(self) -> None:
        """Checks that concrete DEM interfaces do not override elevation methods already supplied by their bases."""

        # 1/ Both concrete interfaces must inherit the common elevation and vertical-reference bases
        assert issubclass(DEM, DEMBase)
        assert issubclass(DEMAccessor, DEMBase)
        assert issubclass(DEMBase, _VerticalReference)

        # 2/ Shared public methods must remain owned by their bases rather than duplicated in either interface
        for parent in (DEMBase, _VerticalReference):
            for name, member in vars(parent).items():
                if name.startswith("_") or not (callable(member) or isinstance(member, property)):
                    continue
                assert name not in DEM.__dict__
                assert name not in DEMAccessor.__dict__


class TestClassVsAccessorConsistencyInherited:
    """
    Test representative ``RasterBase`` behavior inherited by DEM and its Xarray accessor.

    This class tests:
    - Metadata properties that do not read the elevation array,
    - Methods returning coordinates, arrays or rasters,
    - Input and output loading for eager calls,
    - Input and output laziness for Dask calls.

    The complete ``RasterBase`` API is tested in GeoUtils.
    """

    # Run tests for different DEMs
    longyearbyen_path = examples.get_path_test("longyearbyen_ref_dem")

    # Minimal representative subset of inherited attributes
    inherited_attributes = ["crs", "transform", "nodata", "res", "_is_xr"]

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("attr", inherited_attributes)  # type: ignore
    def test_attributes__equality(self, path_dem: str, attr: str) -> None:
        """Test that a minimal subset of inherited attributes of the two objects are exactly the same."""

        # Open
        ds = open_dem(path_dem)
        dem = DEM(path_dem)

        # Get attribute for each object
        output_dem = getattr(dem, attr)
        output_ds = getattr(ds.dem, attr)

        # Assert equality
        if attr != "_is_xr":  # Only attribute that is (purposely) not the same, but the opposite
            assert_output_equal(output_dem, output_ds)
        else:
            assert output_dem != output_ds

    # Minimal representative subset of inherited methods
    inherited_methods_and_kwargs = [
        ("coords", {"grid": True}),  # Metadata-only inherited method
        ("translate", {"xoff": 10.5, "yoff": 5}),  # Raster-returning inherited method
        ("interp_points", {"points": "random", "as_array": True}),  # Array-returning inherited method
        ("reproject", {"crs": CRS.from_epsg(4326)}),  # Raster-returning loading method
        ("set_nodata", {"new_nodata": -10001, "update_array": False, "update_mask": False}),  # In-place method
    ]

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in inherited_methods_and_kwargs])  # type: ignore
    def test_methods__equality(self, path_dem: str, method: str, kwargs: dict[str, Any]) -> None:
        """
        Test that a minimal representative subset of inherited RasterBase methods yield the same outputs
        between a DEM and Xarray dem accessor.
        """

        # Open both objects
        ds = open_dem(path_dem)
        dem = DEM(path_dem)

        args = kwargs.copy()

        # For methods that require knowledge of the data (relative to bounds), create specific inputs
        if "points" in args:
            rng = np.random.default_rng(seed=42)
            ninterp = 10
            res = dem.res
            interp_x = dem.bounds.left + (rng.choice(dem.shape[1], ninterp) + rng.random(ninterp)) * res[0]
            interp_y = dem.bounds.bottom + (rng.choice(dem.shape[0], ninterp) + rng.random(ninterp)) * res[1]
            args.update({"points": (interp_x, interp_y)})

        # Apply method for each class
        output_dem = getattr(dem, method)(**args)
        output_ds = getattr(ds.dem, method)(**args)

        # In-place method
        if method == "set_nodata":
            assert output_dem is None
            assert output_ds is None
            assert_output_equal(dem, ds)
        else:
            assert_output_equal(output_dem, output_ds)

    # Minimal representative subset of inherited methods for loading checks
    inherited_methods_loading_and_kwargs = [
        ("coords", {"grid": True}),  # Metadata-only inherited method, should not load
        ("reproject", {"crs": CRS.from_epsg(4326)}),  # Raster inherited method, should load
    ]

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])
    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in inherited_methods_loading_and_kwargs])
    def test_methods__loading(self, path_dem: str, method: str, kwargs: dict[str, Any]) -> None:
        """
        Test that a minimal subset of inherited RasterBase methods preserve the expected loading behaviour
        between a DEM and Xarray dem accessor.
        """

        # Open both objects
        ds = open_dem(path_dem)
        dem = DEM(path_dem)

        args = kwargs.copy()

        # Apply method for each class
        output_dem = getattr(dem, method)(**args)
        output_ds = getattr(ds.dem, method)(**args)

        # Check using method did or did not load the input DEM or Xarray dataset
        should_input_be_loaded = method not in ["coords"]
        assert dem.is_loaded is should_input_be_loaded
        assert ds._in_memory is should_input_be_loaded

        # In the case of a DEM / DataArray output, check if output is loaded or not
        if isinstance(output_ds, xr.DataArray):
            # coords returns arrays; reproject returns raster-like output and should be loaded here
            assert output_dem.is_loaded
            assert output_ds._in_memory

        # Finally, assert exact equality of outputs
        assert_output_equal(output_dem, output_ds)

    # Minimal representative subset of inherited chunked methods
    inherited_chunked_methods_and_args = [
        ("interp_points", {"points": "random", "as_array": True}),  # Array inherited method
        ("reproject", {"crs": CRS.from_epsg(4326)}),  # Raster inherited method
    ]

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in inherited_chunked_methods_and_args])  # type: ignore
    def test_chunked_methods__loading_laziness(self, path_dem: str, method: str, kwargs: dict[str, Any]) -> None:
        """
        Test that a minimal subset of inherited chunked methods preserve loading and laziness.

        They should yield the exact same output for:
        - Dask backend through Xarray accessor,
        - Multiprocessing backend through DEM class.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # Open lazily with Dask
        ds = open_dem(path_dem, chunks={"band": 1, "x": 25, "y": 25})
        # Open DEM that will be processed using Multiprocessing
        mp_config = MultiprocConfig(chunks=25)
        dem = DEM(path_dem)

        args = kwargs.copy()

        # Special arguments
        if "points" in args:
            rng = np.random.default_rng(seed=42)
            ninterp = 10
            res = dem.res
            interp_x = dem.bounds.left + (rng.choice(dem.shape[1], ninterp) + rng.random(ninterp)) * res[0]
            interp_y = dem.bounds.bottom + (rng.choice(dem.shape[0], ninterp) + rng.random(ninterp)) * res[1]
            args.update({"points": (interp_x, interp_y)})

        # Apply method for each
        output_dem = getattr(dem, method)(**args, mp_config=mp_config)
        output_ds = getattr(ds.dem, method)(**args)

        # For a raster-type output
        if isinstance(output_dem, DEM):

            # 1/ For Dask object: both inputs and outputs should be unloaded + lazy, and compute
            assert not ds._in_memory
            assert isinstance(ds.data, da.Array)
            assert ds.data.chunks is not None

            assert not output_ds._in_memory
            assert isinstance(output_ds.data, da.Array)
            assert output_ds.data.chunks is not None

            output_ds = output_ds.compute()
            assert isinstance(output_ds.data, np.ndarray)
            assert output_ds._in_memory

            # 2/ For Multiprocessing, output remains unloaded
            assert not dem.is_loaded
            assert not output_dem.is_loaded

        # For an array-type output
        elif isinstance(output_dem, np.ndarray):

            # 1/ For Dask object: input and output should be unloaded + lazy, and compute
            assert not ds._in_memory
            assert isinstance(ds.data, da.Array)
            assert ds.data.chunks is not None

            assert isinstance(output_ds, da.Array)
            assert output_ds.chunks is not None

            output_ds = output_ds.compute()
            assert isinstance(output_ds, np.ndarray)

            # 2/ For Multiprocessing, input remains unloaded
            assert not dem.is_loaded
            assert isinstance(output_dem, np.ndarray)

        # Check outputs are the same
        assert_output_equal(output_dem, output_ds, use_allclose=True)


class TestClassVsAccessorConsistencyDEMBase:
    """
    Test the ``DEMBase`` and vertical-reference API through DEM and its Xarray accessor.

    This class tests:
    - Shared properties for equal values and expected loading,
    - Shared methods for equal outputs and expected loading,
    - Class constructors for equal DEM and DataArray results,
    - Automatic coverage of every public method using ``NeedsTestError``,
    - Chunked methods for exact results and unchanged source laziness.

    Comparisons account for masked arrays in DEM and NaN arrays in Xarray. Uncertainty methods are compared in
    tests/test_uncertainty/test_propagation.py.
    """

    # Run tests for different DEMs
    longyearbyen_path = examples.get_path_test("longyearbyen_ref_dem")

    # Get all public properties and methods shared by the DEM interfaces to retain coverage through API changes
    # Inherited RasterBase methods are tested above; _VerticalReference contains the VCRS API shared with EPCBase.
    elevation_bases = (DEMBase, _VerticalReference)
    properties = [
        name
        for base in elevation_bases
        for name, member in vars(base).items()
        if not name.startswith("_")
        and isinstance(member, property)
        and not getattr(member, "__isabstractmethod__", False)
    ]
    methods = [
        name
        for base in elevation_bases
        for name, member in vars(base).items()
        if not name.startswith("_")
        and not isinstance(member, property)
        and callable(member)
        and not getattr(member, "__isabstractmethod__", False)
    ]

    # List of properties that WILL load the input dataset (only one does, the data itself, if DEMBase defines one)
    properties_input_load = ["data"]

    # List of DEM-specific methods that WILL NOT load the input dataset
    methods_input_noload: list[str] = ["set_vcrs"]

    # List of DEM-specific methods that WILL NOT load the input for certain arguments
    methods_input_noload_allowed_args: dict[str, Any] = {}

    # List of DEM-specific methods that WILL NOT load the output dataset
    methods_output_noload: list[str] = ["set_vcrs"]

    # List of DEM-specific methods that WILL NOT LOAD the output for certain arguments
    methods_output_noload_allowed_args: dict[str, Any] = {}

    # Uncertainty methods are covered in tests/test_uncertainty/test_propagation.py
    methods_tested_separately = [
        "estimate_error_structure",
        "estimate_uncertainty",
    ]

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("prop", properties)  # type: ignore
    def test_properties__equality_and_loading(self, path_dem: str, prop: str) -> None:
        """
        Test that DEMBase-specific properties are exactly equal between a DEM and DataArray using the "dem" accessor,
        and if they do not load the dataset or not.
        """

        # Open
        ds = open_dem(path_dem)
        dem = DEM(path_dem)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get attribute for each object
        output_dem = getattr(dem, prop)
        output_ds = getattr(ds.dem, prop)

        # Assert equality
        if prop == "_is_xr":  # Only attribute that is (purposely) not the same, but the boolean opposite
            assert output_dem != output_ds
        else:
            assert_output_equal(output_dem, output_ds)

        # Check getting attribute did not (or did) load the DEM or Xarray dataset
        should_input_be_loaded = prop in self.properties_input_load
        assert dem.is_loaded is should_input_be_loaded
        assert ds._in_memory is should_input_be_loaded

    # Test DEMBase-specific methods
    methods_and_kwargs = [
        # 1. Will load, not inplace
        ("to_vcrs", {"vcrs": "EGM96", "force_source_vcrs": "Ellipsoid"}),
        ("coregister_3d", {"coreg_method": coreg.VerticalShift(subsample=50), "random_state": 42}),
        ("slope", {}),
        ("aspect", {}),
        ("hillshade", {}),
        ("curvature", {}),
        ("profile_curvature", {}),
        ("planform_curvature", {}),
        ("tangential_curvature", {}),
        ("flowline_curvature", {}),
        ("min_curvature", {}),
        ("max_curvature", {}),
        ("topographic_position_index", {}),
        ("terrain_ruggedness_index", {}),
        ("roughness", {}),
        ("rugosity", {}),
        ("fractal_roughness", {}),
        ("texture_shading", {}),
        ("get_terrain_attribute", {"attribute": ["slope", "aspect"]}),
        ("to_pointcloud", {}),
        # 2. Inplace, will not load
        ("set_vcrs", {"new_vcrs": "EGM96"}),
    ]

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("method, kwargs", methods_and_kwargs, ids=[m for m, _ in methods_and_kwargs])
    def test_methods__equality_and_loading(self, path_dem: str, method: str, kwargs: dict[str, Any]) -> None:
        """
        Test that the DEMBase-specific method output and loading mechanism of the two objects are exactly the same
        between a DEM and Xarray dem accessor.
        """

        # Open both objects
        ds = open_dem(path_dem)
        dem = DEM(path_dem)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        args = kwargs.copy()

        # Apply method for each class
        outputs = []
        for source in (dem, ds.dem):
            if method == "coregister_3d":
                args["reference_elev"] = source.copy()
            outputs.append(getattr(source, method)(**args))
        output_dem, output_ds = outputs

        # Determine if operation was in-place or not
        inplace = method in ["set_vcrs"]

        # Native operations retain the DEM/EPC interfaces while accessor operations retain Xarray/Pandas interfaces
        if isinstance(output_dem, Raster):
            assert isinstance(output_dem, DEM)
            assert isinstance(output_ds, xr.DataArray)
        elif isinstance(output_dem, EPC):
            assert isinstance(output_ds, pd.DataFrame)
        elif isinstance(output_dem, list):
            assert all(isinstance(output, DEM) for output in output_dem)
            assert all(isinstance(output, xr.DataArray) for output in output_ds)

        # If yes, outputs should be None, and we'll check loading behaviour for inputs as if they were outputs
        if inplace:
            assert output_dem is None
            assert output_ds is None
            output_ds = ds
            output_dem = dem
        # If no, we check input status
        else:
            # Check using method did or did not load the input DEM or Xarray dataset, following expected values
            should_input_be_loaded = should_be_loaded(
                method=method,
                args=args,
                noload=self.methods_input_noload,
                noload_allowed_args=self.methods_input_noload_allowed_args,
            )
            assert dem.is_loaded is should_input_be_loaded
            assert ds._in_memory is should_input_be_loaded

        # In the case of a DEM / DataArray output, check if output is loaded or not
        # (for in-place methods, we now check the mutated input objects)
        if isinstance(output_ds, xr.DataArray):
            should_output_be_loaded = should_be_loaded(
                method=method,
                args=args,
                noload=self.methods_output_noload,
                noload_allowed_args=self.methods_output_noload_allowed_args,
            )
            assert output_dem.is_loaded is should_output_be_loaded
            assert output_ds._in_memory is should_output_be_loaded

        # Finally, assert exact equality of outputs
        # (in case of DEM; this will load all the data, so has to come at the end)
        assert_output_equal(output_dem, output_ds)

    class_methods_and_kwargs = [
        (
            "from_array",
            {
                "data": np.ones((5, 5)),
                "transform": DEM.from_array(
                    data=np.ones((5, 5)),
                    transform=(1, 0, 0, 0, -1, 5),
                    crs=CRS.from_epsg(4326),
                ).transform,
                "crs": CRS.from_epsg(4326),
                "nodata": -9999,
                "tags": {"metadata": "test"},
                "area_or_point": "Point",
            },
        ),
    ]

    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in class_methods_and_kwargs])
    def test_classmethods__equality(self, method: str, kwargs: dict[str, Any]) -> None:
        """Test class methods output exactly the same objects. Loading always happens for class methods."""

        # Accessor only uses this internally, but we expose it as a class method anyway
        output_dem = getattr(DEM, method)(**kwargs)
        output_ds = getattr(DEMAccessor, method)(**kwargs)

        assert isinstance(output_dem, DEM)
        assert isinstance(output_ds, xr.DataArray)
        assert_output_equal(output_dem, output_ds)

    def test_methods__test_coverage(self) -> None:
        """Checks that every public shared DEM elevation method is assigned to an explicit test."""

        # Compare the assigned tests with the automatically discovered shared API
        methods_1 = [m[0] for m in self.methods_and_kwargs]
        methods_2 = [m[0] for m in self.class_methods_and_kwargs]
        tested_methods = methods_1 + methods_2 + self.methods_tested_separately
        list_missing = [method for method in self.methods if method not in tested_methods]

        if len(list_missing) != 0:
            raise NeedsTestError(f"Shared DEM elevation methods not covered by tests: {list_missing}")

    chunked_methods_and_args = (
        ("to_vcrs", {"vcrs": "EGM96", "force_source_vcrs": "Ellipsoid"}),
        ("coregister_3d", {"coreg_method": coreg.VerticalShift(subsample=50), "random_state": 42}),
        ("slope", {}),
        ("aspect", {}),
        ("hillshade", {}),
        ("curvature", {}),
        ("profile_curvature", {}),
        ("planform_curvature", {}),
        ("tangential_curvature", {}),
        ("flowline_curvature", {}),
        ("max_curvature", {}),
        ("min_curvature", {}),
        ("texture_shading", {}),
        ("topographic_position_index", {}),
        ("terrain_ruggedness_index", {}),
        ("roughness", {}),
        ("rugosity", {}),
        ("fractal_roughness", {}),
        ("get_terrain_attribute", {"attribute": ["slope", "aspect"]}),
    )

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])  # type: ignore
    @pytest.mark.parametrize("method, kwargs", chunked_methods_and_args, ids=[m for m, _ in chunked_methods_and_args])
    def test_chunked_methods__equality_loading_laziness(
        self, path_dem: str, method: str, kwargs: dict[str, Any]
    ) -> None:
        """
        Test that DEMBase-specific chunked methods have the exact same output, loading mechanism and laziness.

        They should yield the exact same output for:
        - In-memory,
        - Dask backend through Xarray accessor,
        - Multiprocessing backend through DEM class.

        Dask array should remain delayed before compute, and Multiprocessing output remains unloaded.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # Open lazily with Dask
        ds = open_dem(path_dem, chunks={"band": 1, "x": 25, "y": 25})
        # Open DEM that will be processed using Multiprocessing
        mp_config = MultiprocConfig(chunks=25)
        dem = DEM(path_dem)
        # Open and load both DataArray/DEM with NumPy
        ds2 = open_dem(path_dem)
        ds2.load()
        dem2 = DEM(path_dem)
        dem2.load()

        args = kwargs.copy()

        # Apply method for each
        outputs = []
        for source, config in ((dem, mp_config), (ds.dem, None), (dem2, None), (ds2.dem, None)):
            if method == "coregister_3d":
                args["reference_elev"] = source.copy()
            outputs.append(getattr(source, method)(**args, mp_config=config))
        output_dem, output_ds, output_dem2, output_ds2 = outputs

        # For a raster-type output
        if isinstance(output_dem, Raster):

            # 1/ For Dask object: both inputs and outputs should be unloaded + lazy, and compute
            # Input
            assert not ds._in_memory
            assert isinstance(ds.data, da.Array)
            assert ds.data.chunks is not None
            # Output
            assert not output_ds._in_memory
            assert isinstance(output_ds.data, da.Array)
            assert output_ds.data.chunks is not None
            # Output computes successfully, and is then loaded in memory
            output_ds = output_ds.compute()
            assert isinstance(output_ds.data, np.ndarray)
            assert output_ds._in_memory

            # 2/ For Multiprocessing, same for loading
            assert not dem.is_loaded
            assert not output_dem.is_loaded

            # 3/ For non-Dask array, both should be loaded
            assert ds2._in_memory
            assert isinstance(ds2.data, np.ndarray)
            assert output_ds2._in_memory
            assert isinstance(output_ds2.data, np.ndarray)

            # 4/ For DEM, same
            assert dem2.is_loaded
            assert output_dem2.is_loaded

        # For an array-type output
        elif isinstance(output_dem, np.ndarray):

            # 1/ For Dask object: both inputs and outputs should be unloaded + lazy, and compute
            # Input
            assert not ds._in_memory
            assert isinstance(ds.data, da.Array)
            assert ds.data.chunks is not None
            # Output
            assert isinstance(output_ds, da.Array)
            assert output_ds.chunks is not None
            # Output computes successfully, and is then loaded in memory
            output_ds = output_ds.compute()
            assert isinstance(output_ds, np.ndarray)

            # 2/ For Multiprocessing, same for loading
            assert not dem.is_loaded
            assert isinstance(output_dem, np.ndarray)

            # 3/ For non-Dask array, both should be loaded
            assert ds2._in_memory
            assert isinstance(ds2.data, np.ndarray)
            assert isinstance(output_ds2, np.ndarray)

            # 4/ For DEM, same
            assert dem2.is_loaded
            assert isinstance(output_dem2, np.ndarray)

        # For a list of raster-type outputs
        elif isinstance(output_dem, list):

            # Preserve the native or Xarray interface for every requested terrain output
            assert all(isinstance(output, DEM) for output in output_dem)
            assert all(isinstance(output, DEM) for output in output_dem2)
            assert all(isinstance(output, xr.DataArray) for output in output_ds)
            assert all(isinstance(output, xr.DataArray) for output in output_ds2)

            # Dask output may be a list of delayed raster-like objects
            assert not ds._in_memory
            assert not dem.is_loaded

            output_ds = [out.compute() if hasattr(out, "compute") else out for out in output_ds]

            assert ds2._in_memory
            assert dem2.is_loaded

        # Check all outputs are exactly the same
        assert_output_equal(output_dem, output_ds)
        assert_output_equal(output_dem, output_dem2, strict_masked=False)
        assert_output_equal(output_dem, output_ds2)

# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

test for workflow class
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import xdem
from xdem.workflows import toolbox

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

pytest.importorskip("cerberus")

import yaml  # type: ignore  # noqa


def test_return_info() -> None:
    """Test metadata extraction from a DEM."""

    def compare(info: Dict[str, Any], dem: xdem.DEM) -> None:
        assert info["Driver"] == dem.driver
        assert info["Filename"] == dem.name
        assert info["Number of band"] == dem.bands
        assert info["Data types"] == str(dem.dtype)
        assert info["Nodata Value"] == (float(dem.nodata) if dem.nodata is not None else None)
        assert info["Pixel interpretation"] == dem.area_or_point
        assert info["Pixel size"] == dem.res
        assert info["Width"] == dem.width
        assert info["Height"] == dem.height
        assert info["Transform"] == dem.transform
        assert info["Bounds"] == dem.bounds

    dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
    print(dem.nodata)
    compare(toolbox.return_info(dem), dem)

    dem = xdem.open_dem(xdem.examples.get_path("longyearbyen_ref_dem"))
    compare(toolbox.return_info(dem.dem), dem.dem)

    dem = xdem.open_dem(xdem.examples.get_path("longyearbyen_ref_dem"), chunks={"x": 200, "y": 200})
    compare(toolbox.return_info(dem.dem), dem.dem)

    dem.dem.set_nodata(None)
    compare(toolbox.return_info(dem.dem), dem.dem)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        pytest.param({"a": 1.2345, "b": 2.9876}, {"a": 1.23, "b": 2.99}, id="test_flat_dict"),
        pytest.param(
            {"a": {"b": 3.14159, "c": [1.6666, 2.5555]}}, {"a": {"b": 3.14, "c": [1.67, 2.56]}}, id="test_nested_dict"
        ),
        pytest.param({"a": (1.23456, 9.8765)}, {"a": (1.23, 9.88)}, id="test_tuple_values"),
        pytest.param(
            {"a": 5.678, "b": "string", "c": 42, "d": [3.3333, "text"]},
            {"a": 5.68, "b": "string", "c": 42, "d": [3.33, "text"]},
            id="test_mixed_types",
        ),
        pytest.param({"a": np.float64(2.71828)}, {"a": 2.72}, id="test_numpy_float"),
    ],
)
def test_floats_process(inputs: Any, expected: Any) -> None:
    """Test floats_process function"""
    assert toolbox.floats_process(inputs) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            {"a": 1.2345, "nested": [np.float64(2.5), (3.14159, 42, None)]},
            {"a": "1.234", "nested": ["2.500", ("3.142", 42, None)]},
        ),
        (1.25, "1.250"),
        ("text", "text"),
    ],
)
def test_recursive_format_values(value: Any, expected: Any) -> None:
    assert toolbox.recursive_format_values(value) == expected


@pytest.mark.parametrize("dask", [False, True])
@pytest.mark.parametrize("with_right", [False, True])
def test_generate_graph(tmp_path: Path, dask: bool, with_right: bool) -> None:
    """Generate a plot with one or two DEMs."""
    pytest.importorskip("matplotlib")
    path = xdem.examples.get_path_test("longyearbyen_tba_dem")
    dem = xdem.open_dem(path, chunks={"x": 50, "y": 50}).dem if dask else xdem.DEM(path)
    out = tmp_path / "test_generate_graph.png"
    toolbox.generate_plot(
        dem,
        title="Test graph",
        file_path=str(out),
        dem_right=dem if with_right else None,
        title_dem_right="Right" if with_right else None,
    )
    assert out.is_file() and out.stat().st_size > 0


@pytest.mark.parametrize("dask", [False, True])
def test_generate_plot_with_profiles(tmp_path: Path, dask: bool) -> None:
    """Generate the profile plot from a DEM."""
    pytest.importorskip("matplotlib")
    path = xdem.examples.get_path_test("longyearbyen_tba_dem")
    dem = xdem.open_dem(path, chunks={"x": 50, "y": 50}).dem if dask else xdem.DEM(path)
    out = tmp_path / "profiles.png"
    toolbox.generate_plot_with_profiles(dem, title="Profiles", file_path=str(out))
    assert out.is_file() and out.stat().st_size > 0


@pytest.mark.parametrize("attributes", [["slope"], ["slope", "aspect"]])
@pytest.mark.parametrize("export_tif", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("dask", [False, True])
def test_generate_terrain_attributes(
    tmp_path: Path, attributes: list[str], export_tif: bool, as_dict: bool, dask: bool
) -> None:
    """Generate terrain attribute plots and optional rasters."""
    pytest.importorskip("matplotlib")
    path = xdem.examples.get_path_test("longyearbyen_tba_dem")
    dem = xdem.open_dem(path, chunks={"x": 50, "y": 50}).dem if dask else xdem.DEM(path)

    (tmp_path / "plots").mkdir()
    (tmp_path / "rasters").mkdir()
    config = {attribute: {} for attribute in attributes} if as_dict else attributes  # type: ignore
    toolbox.generate_terrain_attributes(
        dem,
        attributes,
        tmp_path,
        config,  # type: ignore
        proj_crs=True,
        proj_crs_value=None,
        export_tif=export_tif,
    )  # type: ignore
    out = tmp_path / "plots" / "terrain_attributes_map.png"
    assert out.is_file() and out.stat().st_size > 0
    for attribute in attributes:
        assert (tmp_path / "rasters" / f"{attribute}.tif").is_file() is export_tif


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("count", 42.0, "42"),
        ("smt_COUNT", 42.0, "42"),
        ("percentage", 87.456, "87.46%"),
        ("smt_PERCENTAGE", 87.456, "87.46%"),
        ("mean", -12.345678, "-12.346"),
        (None, -12.345678, "-12.346"),
        ("mean", 0.0001, "1.e-04"),
        (None, 0.001, "0.001"),
    ],
)
def test_format_values_stats(key: str | None, value: float, expected: str) -> None:
    assert toolbox.format_values_stats(value, key) == expected


def test_save_stat_as_csv(get_topo_inputs_config_list: dict[str, Any], tmp_path: Path) -> None:
    """
    Test save_stat_as_csv function
    """

    data = {"a": 1.2345, "b": 2.9876}
    csv_path = tmp_path / "test_save_stat_as_csv_stats.csv"

    toolbox.save_stat_as_csv(data, csv_path)
    assert csv_path.exists()

    output_csv = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(output_csv, pd.DataFrame([data]))

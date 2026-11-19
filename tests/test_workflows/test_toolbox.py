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

# mypy: disable-error-code=no-untyped-def
import csv
import re
from pathlib import Path

import geoutils as gu
import numpy as np
import pytest
import xarray as xr

import xdem
from xdem.workflows.accuracy import Accuracy
from xdem.workflows.schemas import (
    COMPLETE_CONFIG_ACCURACY,
    COMPLETE_CONFIG_TOPO,
    MIN_STATS,
    TERRAIN_ATTRIBUTES_DEFAULT,
)
from xdem.workflows.topo import Topo
from xdem.workflows.workflows import Workflows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

pytest.importorskip("cerberus")

import yaml  # type: ignore  # noqa

from xdem.workflows import toolbox


def test_return_info() -> None:
    """Test metadata extraction from a DEM."""

    def compare(info, dem):
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
def test_floats_process(inputs, expected):
    """Test floats_process function"""
    assert toolbox.floats_process(inputs) == expected


@pytest.mark.parametrize("chunks", [None, {"band": 1, "x": 50, "y": 50}])
def test_generate_graph(get_topo_inputs_config_list, tmp_path, chunks):
    """
    Test generate_plot function
    """
    dem = xdem.open_dem(xdem.examples.get_path_test("longyearbyen_tba_dem"), chunks=chunks)
    filename = "test_generate_graph"
    title = "Test graph"

    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    workflows.create_output_dir()
    workflows.generate_plot(dem, filename=filename, title=title)
    out = tmp_path / "plots" / f"{filename}.png"
    assert out.exists()

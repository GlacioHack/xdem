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

import geoutils as gu
import numpy as np
import pytest
import yaml  # type: ignore

import xdem
from xdem.workflows.topo import Topo
from xdem.workflows.workflows import Workflows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

pytest.importorskip("weasyprint")

def test_workflows_init_wrong_config():
    """
    Test workflows class initialization
    """
    user_config = 2
    with pytest.raises(ValueError, match="The configuration should be provided either as a path"):
        _ = Topo(user_config)  # type: ignore


def test_workflows_init(pipeline_topo, get_topo_inputs_config, tmp_path):
    """
    Test workflows class initialization
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)

    assert isinstance(workflows, Workflows)
    pipeline_gt = pipeline_topo
    pipeline_gt["outputs"] = {"path": str(tmp_path), "level": 1}
    assert workflows.config == pipeline_topo
    assert workflows.level == 1
    assert workflows.outputs_folder == tmp_path
    assert workflows.outputs_folder.exists()
    for folder in ["plots", "rasters", "tables"]:
        assert workflows.outputs_folder.joinpath(folder).exists()
    assert workflows.outputs_folder.joinpath("used_config.yaml").exists()
    assert workflows.dico_to_show == [
        (
            "Information about inputs",
            {
                "reference_elev": {
                    "path_to_elev": xdem.examples.get_path("longyearbyen_tba_dem"),
                    "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
                    "from_vcrs": None,
                    "to_vcrs": None,
                    "downsample": 1,
                }
            },
        )
    ]


@pytest.mark.parametrize(
    "input_data,expected",
    [
        ({"a": 1, "b": None, "c": 3}, {"a": 1, "c": 3}),
        ({"a": 1, "statistics": None, "c": 3}, {"a": 1, "statistics": None, "c": 3}),
        ({"a": {"x": None, "y": 2}, "b": None}, {"a": {"y": 2}}),
        ([1, None, 2, None, 3], [1, 2, 3]),
        ([{"a": 1, "b": None}, {"c": None}, None, {"d": 4}], [{"a": 1}, {}, {"d": 4}]),
        (
            {"a": [1, None, {"b": None, "c": 3}], "d": None, "e": {"f": None, "g": [None, 7]}},
            {"a": [1, {"c": 3}], "e": {"g": [7]}},
        ),
        ({}, {}),
        ([], []),
        ({"a": 1, "b": [2, 3], "c": {"d": 4}}, {"a": 1, "b": [2, 3], "c": {"d": 4}}),
    ],
)
def test_remove_none_cases(get_topo_inputs_config, tmp_path, input_data, expected):
    """
    Test remove_none from dictionary
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    assert workflows.remove_none(input_data) == expected


def test_load_config(get_topo_inputs_config, tmp_path):
    """
    Test load_config function
    """
    cfg = get_topo_inputs_config

    # Succeed
    with open(tmp_path / "temp_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    workflows = Topo(str(tmp_path / "temp_config.yaml"))
    assert workflows.load_config() == cfg

    # Fail
    with pytest.raises(FileNotFoundError, match=f"{tmp_path}/tempconfig.yaml does not exist"):
        _ = Topo(str(tmp_path / "tempconfig.yaml"))


def test_generate_graph(get_topo_inputs_config, tmp_path):
    """
    Test generate_plot function
    """
    dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    filename = "test_generate_graph"
    title = "Test graph"

    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)

    workflows.generate_plot(dem, filename=filename, title=title)
    out = tmp_path / "plots" / f"{filename}.png"
    assert out.exists()


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
def test_floats_process(get_topo_inputs_config, tmp_path, inputs, expected):
    """
    Test floats_process function
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)

    assert workflows.floats_process(inputs) == expected


def test_save_stat_as_csv(get_topo_inputs_config, tmp_path):
    """
    Test save_stat_as_csv function
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)

    data = {"a": 1.2345, "b": 2.9876}
    title = "test_save_stat_as_csv"
    workflows.save_stat_as_csv(data, title)
    out = tmp_path / "tables" / f"{title}_stats.csv"
    assert out.exists()

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        final_dict = list(reader)

    assert final_dict == [{"a": "1.2345", "b": "2.9876"}]


@pytest.mark.parametrize(
    "from_vcrs, to_vcrs",
    [
        [None, None],
        [None, "EGM96"],
        ["EGM96", None],
        [None, "Ellipsoid"],
        ["Ellipsoid", None],
        ["EGM96", "EGM96"],
        ["Ellipsoid", "Ellipsoid"],
        ["EGM96", "Ellipsoid"],
        ["Ellipsoid", "EGM96"],
    ],
)
def test_load_dem(get_dem_config, from_vcrs, to_vcrs):
    """
    Test load_dem function
    """
    config_dem = get_dem_config
    config_dem["from_vcrs"] = from_vcrs
    config_dem["to_vcrs"] = to_vcrs
    input_dem = xdem.DEM(config_dem["path_to_elev"])
    mean_before = np.nanmean(input_dem)

    if from_vcrs is None and from_vcrs != to_vcrs:
        # if no input VRCS but a to_vcrs is given
        with pytest.raises(ValueError, match="corresponding DEM does not have a current VCRS"):
            Workflows.load_dem(config_dem)

    else:
        output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem)
        mean_after = np.nanmean(output_dem)

        # Check output_dem vcrs reference
        if to_vcrs == "EGM96" or (to_vcrs is None and from_vcrs == "EGM96"):
            assert output_dem.vcrs_name == "EGM96 height"
        elif to_vcrs == "Ellipsoid" or (to_vcrs is None and from_vcrs == "Ellipsoid"):
            assert output_dem.vcrs == "Ellipsoid"
        else:
            assert output_dem.vcrs is None

        # Check output_dem
        if from_vcrs == to_vcrs:
            assert output_dem.raster_equal(input_dem)

        # About 32 meters of difference in Svalbard between EGM96 geoid and ellipsoid
        if to_vcrs == "Ellipsoid" and from_vcrs == "EGM96":
            assert mean_after - mean_before == pytest.approx(32, rel=0.1)

        if to_vcrs == "EGM96" and from_vcrs == "Ellipsoid":
            assert mean_after - mean_before == pytest.approx(-32, rel=0.1)

        # Other outputs
        assert mask_path == config_dem["path_to_mask"]
        mask = gu.Vector(mask_path)
        assert inlier_mask == ~mask.create_mask(input_dem)

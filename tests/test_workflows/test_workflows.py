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
from pathlib import Path

import geoutils as gu
import numpy as np
import pytest

import xdem
from xdem.workflows.accuracy import Accuracy
from xdem.workflows.schemas import (
    COMPLETE_CONFIG_ACCURACY,
    MIN_STATS,
    TERRAIN_ATTRIBUTES_DEFAULT,
)
from xdem.workflows.topo import Topo
from xdem.workflows.workflows import Workflows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

pytest.importorskip("cerberus")

import yaml  # type: ignore  # noqa


def test_workflows_init_wrong_config():
    """
    Test workflows class initialization
    """
    user_config = 2
    with pytest.raises(ValueError, match="The configuration should be provided either as a path"):
        _ = Topo(user_config)  # type: ignore


@pytest.mark.parametrize("level", [1, 2])
def test_workflows_init(get_topo_inputs_config_list, tmp_path, level):
    """
    Test workflows class initialization
    """
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    workflows = Topo(user_config)
    workflows.run()

    assert isinstance(workflows, Workflows)
    assert isinstance(workflows, Topo)


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
def test_remove_none_cases(get_topo_inputs_config_list, tmp_path, input_data, expected):
    """
    Test remove_none from dictionary
    """
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    assert workflows.remove_none(input_data) == expected


def test_load_config(get_topo_inputs_config_list, tmp_path):
    """
    Test load_config function
    """
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}

    # Succeed
    with open(tmp_path / "temp_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(user_config, f, allow_unicode=True, default_flow_style=False)

    workflows = Topo(str(tmp_path / "temp_config.yaml"))
    assert workflows.load_config() == user_config

    # Fail
    with pytest.raises(FileNotFoundError, match=f"{tmp_path}/tempconfig.yaml does not exist"):
        _ = Topo(str(tmp_path / "tempconfig.yaml"))


def test_load_config_none(get_topo_inputs_config_list, get_accuracy_inputs_test, tmp_path):
    """
    Test None values in yaml reading function
    """
    # Change values
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["inputs"][0]["set_vcrs"] = None

    # Read working config
    yaml_str = yaml.dump(user_config, allow_unicode=True)
    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_topo = Topo(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_topo, Workflows)
    assert isinstance(workflow_topo, Topo)
    config_output = workflow_topo.load_config()
    assert config_output["inputs"][0]["set_vcrs"] is None

    # Accuracy workflow

    # Change values
    cfg = get_accuracy_inputs_test
    cfg["inputs"]["reference_elev"]["set_vcrs"] = "None"
    cfg["inputs"]["sampling_grid"] = "None"
    cfg["coregistration"] = {}
    cfg["coregistration"]["process"] = False

    # Read working config
    yaml_str = yaml.dump(cfg, allow_unicode=True)
    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_accuracy = Accuracy(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_accuracy, Workflows)
    assert isinstance(workflow_accuracy, Accuracy)
    config_output = workflow_accuracy.load_config()
    assert config_output["inputs"]["reference_elev"]["set_vcrs"] is None
    assert config_output["inputs"]["sampling_grid"] is None


def test_pipeline_accuracy_default_values(get_accuracy_inputs_test, tmp_path):
    """
    Test valid VCRS function for 'from' and 'to'
    """
    accuracy_config = get_accuracy_inputs_test
    yaml_str = yaml.dump(accuracy_config, allow_unicode=True)

    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_accuracy = Accuracy(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_accuracy, Workflows)
    assert isinstance(workflow_accuracy, Accuracy)
    pipeline_accuracy_test = workflow_accuracy.config

    for elev in ["reference_elev", "to_be_aligned_elev"]:
        input_elev = pipeline_accuracy_test["inputs"][elev]
        input_elev_input = accuracy_config["inputs"][elev]
        assert input_elev["path_to_elev"] == input_elev_input["path_to_elev"]
        if "path_to_mask" in input_elev_input:
            assert input_elev["path_to_mask"] == input_elev_input["path_to_mask"]

        assert "set_vcrs" not in input_elev
        assert input_elev["downsample"] == 1

    assert pipeline_accuracy_test["inputs"]["sampling_grid"] == "reference_elev"

    pipeline_corg = pipeline_accuracy_test["coregistration"]
    assert list(pipeline_corg.keys()) == ["step_one", "process"]
    assert pipeline_corg["step_one"] == COMPLETE_CONFIG_ACCURACY["coregistration"]["step_one"]  # type: ignore
    assert pipeline_accuracy_test["coregistration"]["process"]

    assert pipeline_accuracy_test["statistics"] == COMPLETE_CONFIG_ACCURACY["statistics"]
    assert pipeline_accuracy_test["outputs"] == COMPLETE_CONFIG_ACCURACY["outputs"]


def test_pipeline_topo_default_values(get_topo_inputs_config_list, tmp_path):
    """
    Test valid VCRS function for 'from' and 'to'
    """
    topo_config = dict()
    topo_config["inputs"] = get_topo_inputs_config_list
    yaml_str = yaml.dump(topo_config, allow_unicode=True)
    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_topo = Topo(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_topo, Workflows)
    assert isinstance(workflow_topo, Topo)
    pipeline_topo_test = workflow_topo.config

    assert len(pipeline_topo_test["inputs"]) == len(topo_config["inputs"])
    for k, input_elev in enumerate(pipeline_topo_test["inputs"]):
        assert input_elev["path_to_elev"] == topo_config["inputs"][k]["path_to_elev"]
        if "path_to_mask" in topo_config["inputs"][k]:
            assert input_elev["path_to_mask"] == topo_config["inputs"][k]["path_to_mask"]

        assert "set_vcrs" not in input_elev
        assert "downsample" not in input_elev  # default value not taken in "anyof" schema

    assert pipeline_topo_test["statistics"] == MIN_STATS
    assert pipeline_topo_test["terrain_attributes"] == TERRAIN_ATTRIBUTES_DEFAULT
    assert pipeline_topo_test["outputs"] == {"path": "outputs", "level": 1}


def test_generate_graph(get_topo_inputs_config_list, tmp_path):
    """
    Test generate_plot function
    """
    dem = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
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
def test_floats_process(get_topo_inputs_config_list, tmp_path, inputs, expected):
    """
    Test floats_process function
    """
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)

    assert workflows.floats_process(inputs) == expected


def test_save_stat_as_csv(get_topo_inputs_config_list, tmp_path):
    """
    Test save_stat_as_csv function
    """
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    workflows.create_output_dir()

    data = {"a": 1.2345, "b": 2.9876}
    title = "test_save_stat_as_csv"
    workflows.save_stat_as_csv(data, title)
    out = tmp_path / "tables" / f"{title}_stats.csv"
    assert out.exists()

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        final_dict = list(reader)

    assert final_dict == [{"a": "1.2345", "b": "2.9876"}]


@pytest.mark.parametrize("data", ["longyearbyen_ref_dem", "giza_dem"])
@pytest.mark.parametrize("set_vcrs", [None, "Ellipsoid", "EGM96"])
def test_load_dem(data, set_vcrs):
    config_dem = dict()
    config_dem["path_to_elev"] = xdem.examples.get_path(data)

    if data == "longyearbyen_ref_dem":
        config_dem["path_to_mask"] = "longyearbyen_glacier_outlines"

    config_dem["set_vcrs"] = set_vcrs
    output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem)

    dem = xdem.DEM(config_dem["path_to_elev"])

    # VCRS
    if set_vcrs is None:
        assert output_dem.vcrs == dem.vcrs
    else:
        if dem.vcrs is None:
            dem.set_vcrs(set_vcrs)
        else:
            dem.to_vcrs(set_vcrs, inplace=True)

    # DEM
    assert dem.georeferenced_grid_equal(output_dem)

    # MASK
    if "path_to_mask" in config_dem:
        mask = gu.Vector(mask_path)
        assert inlier_mask.georeferenced_grid_equal(~mask.create_mask(dem))


def test_load_dem_alias():
    """
    Test load_dem function with alias
    """

    # Test with no mask
    config_dem = dict()
    config_dem["path_to_elev"] = "longyearbyen_ref_dem"
    output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem)

    assert output_dem.raster_equal(xdem.DEM(xdem.examples.get_path(config_dem["path_to_elev"])))
    assert inlier_mask is None
    assert mask_path is None

    # Test with mask
    config_dem = dict()
    config_dem["path_to_elev"] = "longyearbyen_tba_dem"
    config_dem["path_to_mask"] = "longyearbyen_glacier_outlines"
    output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem)

    assert output_dem == xdem.DEM(xdem.examples.get_path(config_dem["path_to_elev"]))
    assert inlier_mask == ~gu.Vector(mask_path).create_mask(output_dem)
    assert mask_path == xdem.examples.get_path("longyearbyen_glacier_outlines")

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

import numpy as np
import pytest
import yaml  # type: ignore

import xdem
from xdem.workflows.topo_summary import TopoSummary
from xdem.workflows.workflows import Workflows


def test_workflows_init(pipeline_topo, get_topo_inputs_config, tmp_path):
    """
    Test workflows class initialization
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = TopoSummary(user_config)

    assert isinstance(workflows, Workflows)
    pipeline_gt = pipeline_topo
    pipeline_gt["outputs"] = {"path": str(tmp_path), "level": 1}
    assert workflows.config == pipeline_topo
    assert workflows.level == 1
    assert workflows.outputs_folder == tmp_path
    assert workflows.outputs_folder.exists()
    for folder in ["png", "raster", "csv"]:
        assert workflows.outputs_folder.joinpath(folder).exists()
    assert workflows.outputs_folder.joinpath("used_config.yaml").exists()
    assert workflows.dico_to_show == [
        (
            "Information about inputs",
            {
                "dem": xdem.examples.get_path("longyearbyen_tba_dem"),
                "mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
                "from_vcrs": {"common": "EGM96"},
                "to_vcrs": {"common": "EGM96"},
            },
        )
    ]


def test_load_config(get_topo_inputs_config, tmp_path):
    """
    Test load_config function
    """
    cfg = get_topo_inputs_config

    # Succeed
    with open(tmp_path / "temp_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    workflows = TopoSummary(str(tmp_path / "temp_config.yaml"))
    assert workflows.load_config() == cfg

    # Fail
    with pytest.raises(FileNotFoundError, match=f"{tmp_path}/tempconfig.yaml does not exist"):
        _ = TopoSummary(str(tmp_path / "tempconfig.yaml"))


def test_generate_graph(get_topo_inputs_config, tmp_path):
    """
    Test generate_graph function
    """
    dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    title = "test_generate_graph"

    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = TopoSummary(user_config)

    workflows.generate_graph(dem, title)
    out = tmp_path / "png" / f"{title}.png"
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
    workflows = TopoSummary(user_config)

    assert workflows.floats_process(inputs) == expected


def test_save_stat_as_csv(get_topo_inputs_config, tmp_path):
    """
    Test save_stat_as_csv function
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = TopoSummary(user_config)

    data = {"a": 1.2345, "b": 2.9876}
    title = "test_save_stat_as_csv"
    workflows.save_stat_as_csv(data, title)
    out = tmp_path / "csv" / f"{title}.csv"
    assert out.exists()

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        final_dict = list(reader)

    assert final_dict == [{"a": "1.2345", "b": "2.9876"}]


@pytest.mark.skip(reason="not implemented")
def test_generate_dem():
    """
    Test generate_dem function
    """
    pass

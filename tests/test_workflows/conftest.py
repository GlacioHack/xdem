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
Fixtures for test_workflows
"""

# mypy: disable-error-code=no-untyped-def

import pytest

import xdem


@pytest.fixture()
def get_topo_inputs_config_list():
    """
    Return a list of two inputs for topo workflow
    """
    return [
        {
            "path_to_elev": xdem.examples.get_path_test("longyearbyen_tba_dem"),
            "path_to_mask": xdem.examples.get_path_test("longyearbyen_glacier_outlines"),
        },
        {
            "path_to_elev": xdem.examples.get_path_test("longyearbyen_ref_dem"),
        },
    ]


@pytest.fixture()
def get_topo_config_test(get_topo_inputs_config_list, tmp_path):
    """
    Return a minimal topo config file with one input
    """
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    return user_config


@pytest.fixture()
def get_dem_config():
    """
    Return a full dem configuration
    """
    return {
        "path_to_elev": xdem.examples.get_path_test("longyearbyen_ref_dem"),
        "force_source_nodata": -9999,
        "force_vcrs": "Ellipsoid",
        "path_to_mask": xdem.examples.get_path_test("longyearbyen_glacier_outlines"),
        "downsample": 1,
    }


@pytest.fixture()
def get_accuracy_inputs_test():
    """
    Return a minimal accuracy inputs
    """
    return {
        "inputs": {
            "reference_elev": {
                "path_to_elev": xdem.examples.get_path_test("longyearbyen_ref_dem"),
                "path_to_mask": xdem.examples.get_path_test("longyearbyen_glacier_outlines"),
            },
            "to_be_aligned_elev": {
                "path_to_elev": xdem.examples.get_path_test("longyearbyen_tba_dem"),
            },
        },
    }


@pytest.fixture()
def get_accuracy_config_test(tmp_path, get_accuracy_inputs_test):
    """
    Return a minimal accuracy config file
    """
    user_config = get_accuracy_inputs_test
    user_config["outputs"] = {"path": str(tmp_path)}
    return user_config

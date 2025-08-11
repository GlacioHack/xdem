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
from xdem.workflows import Accuracy
from xdem.workflows.schemas import TERRAIN_ATTRIBUTES_DEFAULT


@pytest.fixture()
def get_topo_inputs_config():
    """
    Return minimal configuration for inputs in topo
    """
    return {
        "inputs": {
            "reference_elev": {
                "path_to_elev": xdem.examples.get_path("longyearbyen_tba_dem"),
                "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            }
        },
    }


@pytest.fixture()
def get_accuracy_inputs_config():
    """
    Return minimal configuration for inputs in accuracy
    """
    return {
        "inputs": {
            "reference_elev": {
                "path_to_elev": xdem.examples.get_path("longyearbyen_ref_dem"),
                "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            },
            "to_be_aligned_elev": {
                "path_to_elev": xdem.examples.get_path("longyearbyen_tba_dem"),
                "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            },
        },
    }


@pytest.fixture()
def get_accuracy_object_with_run(tmp_path):
    """
    Generate classical accuracy object
    """
    user_config = {
        "inputs": {
            "reference_elev": {
                "path_to_elev": xdem.examples.get_path("longyearbyen_ref_dem"),
                "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            },
            "to_be_aligned_elev": {
                "path_to_elev": xdem.examples.get_path("longyearbyen_tba_dem"),
                "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            },
        },
        "outputs": {"path": str(tmp_path)},
    }
    workflows = Accuracy(user_config)
    workflows.run()

    return workflows


@pytest.fixture()
def pipeline_topo():
    """
    Return default configuration for pipeline topo_summary
    """
    return {
        "inputs": {
            "reference_elev": {
                "path_to_elev": xdem.examples.get_path("longyearbyen_tba_dem"),
                "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
                "from_vcrs": "EGM96",
                "to_vcrs": "EGM96",
            }
        },
        "statistics": [
            "mean",
            "median",
            "max",
            "min",
            "sum",
            "sumofsquares",
            "90thpercentile",
            "le90",
            "nmad",
            "rmse",
            "std",
            "standarddeviation",
            "validcount",
            "totalcount",
            "percentagevalidpoints",
        ],
        "terrain_attributes": TERRAIN_ATTRIBUTES_DEFAULT,
        "outputs": {"path": "outputs", "level": 1},
    }


@pytest.fixture()
def list_default_terrain_attributes():
    """
    Return default list of terrain attributes
    """
    return TERRAIN_ATTRIBUTES_DEFAULT

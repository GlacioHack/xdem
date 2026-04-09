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
test for schema files
"""

import logging

import pyproj

# mypy: disable-error-code=no-untyped-def
import pytest

import xdem
from xdem.workflows import schemas
from xdem.workflows.schemas import (
    COMPLETE_CONFIG_ACCURACY,
    MIN_STATS,
    TERRAIN_ATTRIBUTES_DEFAULT,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
pytest.importorskip("cerberus")


def test_validate_base_configuration(get_topo_config_test, get_accuracy_config_test):
    """
    Test validate_base_configuration function
    """
    schemas.validate_configuration(get_topo_config_test, schemas.TOPO_SCHEMA)
    schemas.validate_configuration(get_accuracy_config_test, schemas.ACCURACY_SCHEMA)


def test_wrong_path(get_topo_config_test):
    """
    Test wrong_path function
    """
    topo_config = get_topo_config_test
    topo_config["inputs"][0]["path_to_elev"] = "doesn_t_exist.tif"

    with pytest.raises(ValueError, match="Path does not exist: doesn_t_exist.tif"):
        _ = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)


@pytest.mark.parametrize(
    "new_param_config, expected",
    [
        pytest.param({"outputs": {"level": 3}}, "unallowed value 3", id="outputs_level"),
        pytest.param({"outputs": {"path": 3}}, "must be of string type", id="outputs_path_int"),
        pytest.param(
            {"outputs": {"level": "level_3"}},
            "must be of integer type",
            id="outputs_level_str",
        ),
        pytest.param({"statistics": {"mean": 0.5}}, "must be of list type", id="statistics_dict_in"),
        pytest.param(
            {"statistics": ["wrong_metrics"]},
            "unallowed values",
            id="statistics_wrong_metrics",
        ),
        pytest.param(
            {"terrain_attributes": ["wrong_attr"]},
            "no definitions validate",
            id="terrain_attributes_wrong_attr",
        ),
        pytest.param(
            {"terrain_attributes": {"slope": {"extra_information": 2}}},
            "must be of dict type",
            id="terrain_attributes_dict_attr",
        ),
        pytest.param(
            {
                "inputs": [
                    {
                        "path_to_elev": xdem.examples.get_path_test("longyearbyen_tba_dem"),
                        "downsample": "10",
                    }
                ]
            },
            r"must be of \['integer', 'float'\] type",
            id="downsample as string",
        ),
        pytest.param(
            {
                "inputs": [
                    {
                        "path_to_elev": xdem.examples.get_path_test("longyearbyen_tba_dem"),
                        "downsample": 0,
                    }
                ]
            },
            "min value is 1",
            id="downsample error <1",
        ),
    ],
)
def test_validate_topo_configuration_with_errors(get_topo_config_test, new_param_config, expected):
    """
    Test validation of configuration with errors
    """
    topo_config = get_topo_config_test
    topo_config.update(new_param_config)

    with pytest.raises(ValueError, match=expected):
        _ = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)


@pytest.mark.parametrize(
    "new_param_config, expected",
    [
        pytest.param(
            {"inputs": {"sampling_grid": 3}},
            "must be of string type",
            id="sampling_grid",
        ),
        pytest.param(
            {"inputs": {"sampling_grid": "not_a_dem"}},
            "unallowed value not_a_dem",
            id="sampling_grid",
        ),
    ],
)
def test_validate_topo_coreg_configuration_with_errors(get_accuracy_config_test, new_param_config, expected):
    """
    Test validation of coregistration configuration with errors
    """
    topo_config = get_accuracy_config_test
    topo_config.update(new_param_config)

    with pytest.raises(ValueError, match=expected):
        _ = schemas.validate_configuration(topo_config, schemas.ACCURACY_SCHEMA)


@pytest.mark.parametrize(
    "required,expected_required",
    [
        (True, True),
        (False, False),
    ],
)
def test_required_flag(required, expected_required):
    """
    Test required_flag in coregistration
    """
    schema = schemas.make_coreg_step(required=required)
    assert schema["required"] == expected_required
    assert schema["schema"]["method"]["required"] == expected_required


@pytest.mark.parametrize(
    "default_method,expected_present",
    [
        ("method_a", True),
        (None, False),
    ],
)
def test_default_method_handling(default_method, expected_present):
    """
    Test default method handling for coregistration
    """
    schema = schemas.make_coreg_step(default_method=default_method)
    assert ("default" in schema) == expected_present
    if expected_present:
        assert schema["default"]["method"] == default_method


def test_allowed_methods():
    """
    Test allowed_methods in coregistration
    """
    schema = schemas.make_coreg_step()
    assert schema["schema"]["method"]["allowed"] == schemas.COREG_METHODS


def test_extra_information_is_optional():
    """
    Test extra_information_is_optional in coregistration
    """
    schema = schemas.make_coreg_step()
    assert "extra_information" in schema["schema"]
    assert not schema["schema"]["extra_information"]["required"]


@pytest.mark.parametrize(
    "prefix, vcrs",
    [
        ("from_vcrs", "EGM96"),
        ("from_vcrs", "EGM08"),
        ("from_vcrs", "Ellipsoid"),
        # ("from_vcrs", "no_kv_arcgp-2006-sk.tif"),
        ("from_vcrs", 4326),
    ],
)
def test_valid_from_vcrs(get_accuracy_config_test, pipeline_topo, prefix, vcrs):
    """
    Test valid VCRS function for 'from' and 'to'
    """
    accuracy_config = get_accuracy_config_test
    accuracy_config["inputs"]["reference_elev"].update({prefix: vcrs})
    accuracy_config["inputs"]["reference_elev"].update({prefix: vcrs})
    accuracy_config["inputs"]["to_be_aligned_elev"].update({prefix: vcrs})

    """for input_elev in pipeline_test["inputs"]:
        assert input["path_to_elev"] == accuracy_config["path_to_elev"]
        if "path_to_mask" in input:
            assert input["path_to_mask"] == accuracy_config["path_to_mask"]
        assert input[prefix] == vcrs
        assert input["to_vcrs"] == None
        assert input["downsample"] == 1

    assert pipeline_test["statistics"] == MIN_STATS
    assert pipeline_test["terrain_attributes"] == TERRAIN_ATTRIBUTES_DEFAULT
    assert pipeline_test["outputs"] == {"path": "outputs", "level": 1}"""


def test_pipeline_topo_default_values(get_topo_inputs_config_list):
    """
    Test valid VCRS function for 'from' and 'to'
    """

    topo_config = dict()
    topo_config["inputs"] = get_topo_inputs_config_list
    pipeline_topo_test = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)

    assert len(pipeline_topo_test["inputs"]) == len(topo_config["inputs"])
    for k, input_elev in enumerate(pipeline_topo_test["inputs"]):
        assert input_elev["path_to_elev"] == topo_config["inputs"][k]["path_to_elev"]
        if "path_to_mask" in topo_config["inputs"][k]:
            assert input_elev["path_to_mask"] == topo_config["inputs"][k]["path_to_mask"]
        assert input_elev["from_vcrs"] is None
        assert input_elev["to_vcrs"] is None
        assert input_elev["downsample"] == 1

    assert pipeline_topo_test["statistics"] == MIN_STATS
    assert pipeline_topo_test["terrain_attributes"] == TERRAIN_ATTRIBUTES_DEFAULT
    assert pipeline_topo_test["outputs"] == {"path": "outputs", "level": 1}


def test_pipeline_accuracy_default_values(get_accuracy_inputs_test):
    """
    Test valid VCRS function for 'from' and 'to'
    """
    accuracy_config = get_accuracy_inputs_test
    pipeline_accuracy_test = schemas.validate_configuration(accuracy_config, schemas.ACCURACY_SCHEMA)

    for elev in ["reference_elev", "to_be_aligned_elev"]:
        input_elev = pipeline_accuracy_test["inputs"][elev]
        input_elev_input = accuracy_config["inputs"][elev]
        assert input_elev["path_to_elev"] == input_elev_input["path_to_elev"]
        if "path_to_mask" in input_elev_input:
            assert input_elev["path_to_mask"] == input_elev_input["path_to_mask"]
        assert input_elev["from_vcrs"] is None
        assert input_elev["to_vcrs"] is None
        assert input_elev["downsample"] == 1
    assert pipeline_accuracy_test["inputs"]["sampling_grid"] == "reference_elev"

    assert list(pipeline_accuracy_test["coregistration"].keys()) == ["step_one", "process"]
    """assert (
        pipeline_accuracy_test["coregistration"]["step_one"] == COMPLETE_CONFIG_ACCURACY["coregistration"]["step_one"]
    )"""  # TODO
    assert pipeline_accuracy_test["coregistration"]["process"]
    assert pipeline_accuracy_test["statistics"] == COMPLETE_CONFIG_ACCURACY["statistics"]
    assert pipeline_accuracy_test["outputs"] == COMPLETE_CONFIG_ACCURACY["outputs"]


@pytest.mark.parametrize(
    "vcrs, error",
    [
        ("EGM96", False),
        ("EGM08", False),
        ("Ellipsoid", False),
        # ("to_vcrs", "no_kv_arcgp-2006-sk.tif"),
        (4326, False),
    ],
)
def test_valid_vcrs(get_topo_config_test, vcrs, error):
    """
    Test valid VCRS function for 'from' and 'to'
    """
    topo_config = get_topo_config_test
    topo_config["inputs"][0]["to_vcrs"] = vcrs
    topo_config["inputs"][0]["from_vcrs"] = vcrs
    if error is False:
        pipeline_test = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)
        assert pipeline_test["inputs"][0]["to_vcrs"] == vcrs
        assert pipeline_test["inputs"][0]["from_vcrs"] == vcrs


@pytest.mark.parametrize(
    "wrong_vcrs, error",
    [
        pytest.param(
            "wrong",
            "LoggingError",
            id="wrong_common",
        ),
        pytest.param(
            "wrong.txt",
            "LoggingError",
            id="wrong_proj_grid",
        ),
        pytest.param(
            0000,
            pyproj.exceptions.CRSError,
            id="wrong_epsg_code",
        ),
    ],
)
def test_invalid_vcrs(get_topo_config_test, wrong_vcrs, error, caplog, assert_and_allow_log):
    """
    Test invalid crs
    """
    topo_config = get_topo_config_test
    topo_config["inputs"][0].update({"from_vcrs": wrong_vcrs})

    if error == "LoggingError":
        with caplog.at_level(logging.ERROR):
            _ = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)
        assert_and_allow_log(caplog, level=logging.ERROR, match="'from_vcrs' field is not valid.*")
    else:
        with pytest.raises(error):
            _ = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)


def test_topo_without_terrain_attributes_in_config(get_topo_config_test):
    """
    Test different value for terrain attributes in config
    """
    topo_config = get_topo_config_test
    doc = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)
    assert doc["terrain_attributes"] == schemas.TERRAIN_ATTRIBUTES_DEFAULT

    topo_config = get_topo_config_test
    topo_config["terrain_attributes"] = []
    doc = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)
    assert doc["terrain_attributes"] == []

    topo_config = get_topo_config_test
    topo_config["terrain_attributes"] = ["hillshade", "slope", "max_curvature"]
    doc = schemas.validate_configuration(topo_config, schemas.TOPO_SCHEMA)
    assert doc["terrain_attributes"] == ["hillshade", "slope", "max_curvature"]

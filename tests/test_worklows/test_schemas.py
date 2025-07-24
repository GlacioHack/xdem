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
# mypy: disable-error-code=no-untyped-def
import re

import pytest

from xdem.workflows import schemas


def test_validate_base_configuration(get_topo_inputs_config, get_compare_inputs_config):
    """
    Test validate_base_configuration function
    """
    schemas.validate_configuration(get_topo_inputs_config, schemas.INFO_SCHEMA)
    schemas.validate_configuration(get_compare_inputs_config, schemas.COMPARE_SCHEMA)


def test_wrong_path(get_topo_inputs_config):
    """
    Test wrong_path function
    """
    info_conf = get_topo_inputs_config
    info_conf["inputs"]["dem"] = "doesn_t_exist.tif"
    expected = "User configuration mistakes in 'inputs': [{'dem': ['Path does not exist: doesn_t_exist.tif']}]"

    with pytest.raises(ValueError, match=re.escape(expected)):
        _ = schemas.validate_configuration(info_conf, schemas.INFO_SCHEMA)


@pytest.mark.parametrize(
    "new_param_config, expected",
    [
        pytest.param({"outputs": {"level": 3}}, " 'outputs': [{'level': ['unallowed value 3']}]", id="outputs_level"),
        pytest.param(
            {"outputs": {"path": 3}}, " 'outputs': [{'path': ['must be of string type']}]", id="outputs_path_int"
        ),
        pytest.param(
            {"outputs": {"level": "level_3"}},
            " 'outputs': [{'level': ['must be of integer type']}]",
            id="outputs_level_str",
        ),
        pytest.param({"statistics": {"mean": 0.5}}, " 'statistics': ['must be of list type']", id="statistics_dict_in"),
        pytest.param(
            {"statistics": ["wrong_metrics"]},
            " 'statistics': [\"unallowed values ('wrong_metrics',)\"]",
            id="statistics_wrong_metrics",
        ),
        pytest.param(
            {"terrain_attributes": ["wrong_attr"]},
            " 'terrain_attributes': ['no definitions validate', "
            "{'anyof definition 0': [{0: ['unallowed value wrong_attr']}], "
            "'anyof definition 1': ['must be of dict type']}]",
            id="terrain_attributes_wrong_attr",
        ),
        pytest.param(
            {"terrain_attributes": {"slope": {"extra_information": 2}}},
            " 'terrain_attributes': ['no definitions validate', {'anyof definition 0': ['must be of list type'], "
            "'anyof definition 1': [{'slope': [{'extra_information': ['must be of dict type']}]}]}]",
            id="terrain_attributes_dict_attr",
        ),
    ],
)
def test_validate_info_configuration_with_errors(get_topo_inputs_config, new_param_config, expected):
    """
    Test validation of configuration with errors
    """
    info_conf = get_topo_inputs_config
    info_conf.update(new_param_config)
    info_str = "User configuration mistakes in" + expected

    with pytest.raises(ValueError, match=re.escape(info_str)):
        _ = schemas.validate_configuration(info_conf, schemas.INFO_SCHEMA)


@pytest.mark.parametrize(
    "new_param_config, expected",
    [
        pytest.param(
            {"coregistration": {"sampling_source": 3}},
            " 'coregistration': [{'sampling_source': ['must be of string type']}]",
            id="sampling_source",
        ),
        pytest.param(
            {"coregistration": {"sampling_source": "not_a_dem"}},
            " 'coregistration': [{'sampling_source': ['unallowed value not_a_dem']}]",
            id="sampling_source",
        ),
    ],
)
def test_validate_info_coreg_configuration_with_errors(get_compare_inputs_config, new_param_config, expected):
    """
    Test validation of coregistration configuration with errors
    """
    info_conf = get_compare_inputs_config
    info_conf.update(new_param_config)
    info_str = "User configuration mistakes in" + expected

    with pytest.raises(ValueError, match=re.escape(info_str)):
        _ = schemas.validate_configuration(info_conf, schemas.COMPARE_SCHEMA)


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
        ("from_vcrs", {"common": "EGM96"}),
        ("from_vcrs", {"common": "EGM08"}),
        ("from_vcrs", {"common": "Ellipsoid"}),
        ("from_vcrs", {"proj_grid": "no_kv_arcgp-2006-sk.tif"}),
        ("from_vcrs", {"epsg_code": 4326}),
        ("to_vcrs", {"common": "EGM96"}),
        ("to_vcrs", {"common": "EGM08"}),
        ("to_vcrs", {"common": "Ellipsoid"}),
        ("to_vcrs", {"proj_grid": "no_kv_arcgp-2006-sk.tif"}),
        ("to_vcrs", {"epsg_code": 4326}),
    ],
)
def test_valid_vcrs(get_topo_inputs_config, pipeline_topo, prefix, vcrs):
    """
    Test valid VCRS function for 'from' and 'to'
    """
    info_conf = get_topo_inputs_config
    info_conf["inputs"].update({prefix: vcrs})

    pipeline_test = schemas.validate_configuration(info_conf, schemas.INFO_SCHEMA)
    pipeline_test["inputs"].update({prefix: vcrs})
    pipeline_topo["inputs"].update({prefix: vcrs})
    assert pipeline_topo == pipeline_test


@pytest.mark.parametrize(
    "wrong_vcrs, expected",
    [
        pytest.param(
            4326,
            "must be of dict type",
            id="not_a_dictionary",
        ),
        pytest.param(
            {"common": "EGM96", "epsg_code": 4326},
            "Only one of",
            id="two_keys",
        ),
        pytest.param(
            {"common": "wrong"},
            "Invalid common value",
            id="wrong_common",
        ),
        pytest.param(
            {"proj_grid": 0},
            "proj_grid must be a string path",
            id="wrong_proj_grid",
        ),
        pytest.param(
            {"proj_grid": "wrong.txt"},
            "proj_grid must point to a .tif file",
            id="wrong_proj_grid",
        ),
        pytest.param(
            {"epsg_code": "wrong.txt"},
            "epsg_code must be an integer",
            id="wrong_epsg_code_type",
        ),
        pytest.param(
            {"epsg_code": 0000},
            "Invalid EPSG code",
            id="wrong_epsg_code",
        ),
        pytest.param(
            {"my_crs": 0000},
            "Unknown keys in CRS",
            id="unknown_key",
        ),
    ],
)
def test_invalid_vcrs(get_topo_inputs_config, pipeline_topo, wrong_vcrs, expected):
    """
    Test invalid crs
    """
    info_conf = get_topo_inputs_config
    info_conf["inputs"].update({"from_vcrs": wrong_vcrs})

    with pytest.raises(ValueError, match=expected):
        _ = schemas.validate_configuration(info_conf, schemas.INFO_SCHEMA)

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
import re

import pytest

from xdem.workflows import schemas


def test_validate_base_configuration(get_info_inputs_config, get_compare_inputs_config):
    """ """
    schemas.validate_configuration(get_info_inputs_config, schemas.INFO_SCHEMA)
    schemas.validate_configuration(get_compare_inputs_config, schemas.COMPARE_SCHEMA)


def test_wrong_path(get_info_inputs_config):
    """ """
    info_conf = get_info_inputs_config
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
            " 'terrain_attributes': ['no definitions validate', {'anyof definition 0': [{0: ['unallowed value wrong_attr']}], "
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
def test_validate_info_configuration_with_errors(get_info_inputs_config, new_param_config, expected):
    """ """
    info_conf = get_info_inputs_config
    info_conf.update(new_param_config)
    print(info_conf)
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
def test_validate_info_configuration_with_errors(get_compare_inputs_config, new_param_config, expected):
    """ """
    info_conf = get_compare_inputs_config
    info_conf.update(new_param_config)
    print(info_conf)
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
    schema = schemas.make_coreg_step(default_method=default_method)
    assert ("default" in schema) == expected_present
    if expected_present:
        assert schema["default"]["method"] == default_method


def test_allowed_methods():
    schema = schemas.make_coreg_step()
    assert schema["schema"]["method"]["allowed"] == schemas.COREG_METHODS


def test_extra_information_is_optional():
    schema = schemas.make_coreg_step()
    assert "extra_information" in schema["schema"]
    assert not schema["schema"]["extra_information"]["required"]

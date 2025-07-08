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
import pyproj

# mypy: disable-error-code=no-untyped-def
import pytest

from xdem.workflows import schemas


def test_validate_base_configuration(get_topo_inputs_config, get_diffanalysis_inputs_config):
    """
    Test validate_base_configuration function
    """
    schemas.validate_configuration(get_topo_inputs_config, schemas.TOPO_SUMMARY_SCHEMA)
    schemas.validate_configuration(get_diffanalysis_inputs_config, schemas.DIFF_ANALYSIS_SCHEMA)


def test_wrong_path(get_topo_inputs_config):
    """
    Test wrong_path function
    """
    info_conf = get_topo_inputs_config
    info_conf["inputs"]["reference_elev"]["path_to_elev"] = "doesn_t_exist.tif"

    with pytest.raises(ValueError, match="Path does not exist: doesn_t_exist.tif"):
        _ = schemas.validate_configuration(info_conf, schemas.TOPO_SUMMARY_SCHEMA)


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
    ],
)
def test_validate_info_configuration_with_errors(get_topo_inputs_config, new_param_config, expected):
    """
    Test validation of configuration with errors
    """
    info_conf = get_topo_inputs_config
    info_conf.update(new_param_config)

    with pytest.raises(ValueError, match=expected):
        _ = schemas.validate_configuration(info_conf, schemas.TOPO_SUMMARY_SCHEMA)


@pytest.mark.parametrize(
    "new_param_config, expected",
    [
        pytest.param(
            {"coregistration": {"sampling_grid": 3}},
            "must be of string type",
            id="sampling_grid",
        ),
        pytest.param(
            {"coregistration": {"sampling_grid": "not_a_dem"}},
            "unallowed value not_a_dem",
            id="sampling_grid",
        ),
    ],
)
def test_validate_info_coreg_configuration_with_errors(get_diffanalysis_inputs_config, new_param_config, expected):
    """
    Test validation of coregistration configuration with errors
    """
    info_conf = get_diffanalysis_inputs_config
    info_conf.update(new_param_config)

    with pytest.raises(ValueError, match=expected):
        _ = schemas.validate_configuration(info_conf, schemas.DIFF_ANALYSIS_SCHEMA)


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
        ("to_vcrs", "EGM96"),
        ("to_vcrs", "EGM08"),
        ("to_vcrs", "Ellipsoid"),
        # ("to_vcrs", "no_kv_arcgp-2006-sk.tif"),
        ("to_vcrs", 4326),
    ],
)
def test_valid_vcrs(get_topo_inputs_config, pipeline_topo, prefix, vcrs):
    """
    Test valid VCRS function for 'from' and 'to'
    """
    info_conf = get_topo_inputs_config
    info_conf["inputs"]["reference_elev"].update({prefix: vcrs})

    pipeline_test = schemas.validate_configuration(info_conf, schemas.TOPO_SUMMARY_SCHEMA)
    pipeline_test["inputs"]["reference_elev"].update({prefix: vcrs})
    pipeline_topo["inputs"]["reference_elev"].update({prefix: vcrs})
    assert pipeline_topo == pipeline_test


@pytest.mark.parametrize(
    "wrong_vcrs, error",
    [
        pytest.param(
            "wrong",
            UserWarning,
            id="wrong_common",
        ),
        pytest.param(
            "wrong.txt",
            UserWarning,
            id="wrong_proj_grid",
        ),
        pytest.param(
            0000,
            pyproj.exceptions.CRSError,
            id="wrong_epsg_code",
        ),
    ],
)
def test_invalid_vcrs(get_topo_inputs_config, pipeline_topo, wrong_vcrs, error):
    """
    Test invalid crs
    """
    info_conf = get_topo_inputs_config
    info_conf["inputs"]["reference_elev"].update({"from_vcrs": wrong_vcrs})

    with pytest.raises(error):
        _ = schemas.validate_configuration(info_conf, schemas.TOPO_SUMMARY_SCHEMA)


def test_topo_without_terrain_attributes_in_config(get_topo_inputs_config):
    """
    Test different value for terrain attributes in config
    """
    info_conf = get_topo_inputs_config
    doc = schemas.validate_configuration(info_conf, schemas.TOPO_SUMMARY_SCHEMA)
    assert doc["terrain_attributes"] == schemas.TERRAIN_ATTRIBUTES_DEFAULT

    info_conf = get_topo_inputs_config
    info_conf["terrain_attributes"] = []
    doc = schemas.validate_configuration(info_conf, schemas.TOPO_SUMMARY_SCHEMA)
    assert doc["terrain_attributes"] == []

    info_conf = get_topo_inputs_config
    info_conf["terrain_attributes"] = ["hillshade", "slope", "curvature"]
    doc = schemas.validate_configuration(info_conf, schemas.TOPO_SUMMARY_SCHEMA)
    assert doc["terrain_attributes"] == ["hillshade", "slope", "curvature"]

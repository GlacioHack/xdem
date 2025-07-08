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
Schema constants and validation function
"""
import logging
import os
from typing import Any, Dict
from urllib.error import HTTPError, URLError

from cerberus import Validator

from xdem.vcrs import _vcrs_from_user_input


class CustomValidator(Validator):  # type: ignore
    def _validate_path_exists(self, path_exists: bool, field: str, value: str) -> bool:
        """
        {'type': 'boolean'}
        """
        if path_exists and not os.path.exists(value):
            self._error(field, f"Path does not exist: {value}")
        return True

    def _validate_crs(self, crs: bool, field: str, value: str | int) -> bool:
        """
        {'type': 'boolean'}
        """
        try:
            _vcrs_from_user_input(value)
            return True
        except (ValueError, TypeError, ConnectionResetError, HTTPError, URLError):
            logging.error(f"'{field}' field is not valid. See: https://xdem.readthedocs.io/en/stable/vertical_ref.html")
            return False


INPUTS_DEM = {
    "path_to_elev": {"type": "string", "required": True, "path_exists": True},
    "nodata": {"type": ["integer", "float"], "required": False},
    "path_to_mask": {"type": "string", "required": False, "path_exists": True},
    "from_vcrs": {"type": ["integer", "string"], "required": False, "crs": True, "default": "EGM96"},
    "to_vcrs": {"type": ["integer", "string"], "required": False, "crs": True, "default": "EGM96"},
}

COREG_METHODS = [
    "NuthKaab",
    "DhMinimize",
    "VerticalShift",
    "DirectionalBias",
    "TerrainBias",
]

STATS_METHODS = [
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
]

TERRAIN_ATTRIBUTES_DEFAULT = ["slope", "aspect", "curvature"]

TERRAIN_ATTRIBUTES = [
    "slope",
    "aspect",
    "hillshade",
    "curvature",
    "planform_curvature",
    "profile_curvature",
    "maximum_curvature",
    "terrain_ruggedness_index",
    "topographic_position_index",
    "roughness",
    "rugosity",
    "fractal_roughness",
]


def make_coreg_step(required: bool = False, default_method: str = None) -> Validator.schema:
    """
    Create a coreg schema step to avoid repetition for a step
    :param required: is a step required
    :param default_method: coreg default method
    """
    step_schema = {
        "type": "dict",
        "required": required,
        "schema": {
            "method": {
                "type": "string",
                "allowed": COREG_METHODS,
                "required": True if required else False,
            },
            "extra_information": {"type": "dict", "required": False},
        },
    }
    if default_method:
        step_schema["default"] = {"method": default_method}
    return step_schema


def validate_configuration(user_config: dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the configuration:
    :param user_config: Configuration dict or YAML string
    :param schema: Schema dict for validating configuration
    :return: Completed configuration dictionary
    """
    validator = CustomValidator(schema)
    if not validator.validate(user_config):
        for field, errors in validator.errors.items():
            raise ValueError(f"User configuration mistakes in '{field}': {errors}")

    if "statistics" not in validator.document:
        validator.document["statistics"] = STATS_METHODS

    if "terrain_attributes" not in validator.document and "coregistration" not in validator.document:
        validator.document["terrain_attributes"] = TERRAIN_ATTRIBUTES_DEFAULT

    return validator.document


DIFF_ANALYSIS_SCHEMA = {
    "inputs": {
        "type": "dict",
        "required": True,
        "schema": {
            "reference_elev": {"type": "dict", "schema": INPUTS_DEM, "required": False, "nullable": True},
            "to_be_aligned_elev": {"type": "dict", "schema": INPUTS_DEM, "required": True},
        },
    },
    "outputs": {
        "type": "dict",
        "required": False,
        "default": {"path": "outputs", "level": 1, "output_grid": "reference_elev"},
        "schema": {
            "path": {"type": "string", "required": True, "default": "outputs"},
            "level": {"type": "integer", "default": 1, "required": False, "allowed": [1, 2]},
            "output_grid": {
                "type": "string",
                "allowed": ["reference_elev", "to_be_aligned_elev"],
                "default": "reference_elev",
                "required": False,
            },
        },
    },
    "coregistration": {
        "type": "dict",
        "required": False,
        "default": {"step_one": {"method": "NuthKaab"}, "sampling_grid": "reference_elev"},
        "schema": {
            "step_one": make_coreg_step(default_method="NuthKaab"),
            "step_two": make_coreg_step(required=False),
            "step_three": make_coreg_step(required=False),
            "sampling_grid": {
                "type": "string",
                "allowed": ["reference_elev", "to_be_aligned_elev"],
                "default": "reference_elev",
                "required": False,
            },
            "process": {"type": "boolean", "default": True, "required": False},
        },
    },
    "statistics": {"type": "list", "required": False, "allowed": STATS_METHODS, "nullable": True},
}

TOPO_SUMMARY_SCHEMA = {
    "inputs": {
        "type": "dict",
        "required": True,
        "schema": {
            "reference_elev": {"type": "dict", "schema": INPUTS_DEM, "required": False},
        },
    },
    "statistics": {"type": "list", "required": False, "allowed": STATS_METHODS, "nullable": True},
    "terrain_attributes": {
        "required": False,
        "default": TERRAIN_ATTRIBUTES_DEFAULT,
        "nullable": True,
        "anyof": [
            {
                "type": "list",
                "schema": {
                    "type": "string",
                    "allowed": TERRAIN_ATTRIBUTES,
                },
            },
            {
                "type": "dict",
                "keysrules": {"type": "string", "allowed": TERRAIN_ATTRIBUTES},
                "valuesrules": {
                    "type": "dict",
                    "schema": {
                        "extra_information": {"type": "dict", "required": False},
                    },
                },
            },
        ],
    },
    "outputs": {
        "type": "dict",
        "default": {"path": "outputs", "level": 1},
        "schema": {
            "path": {"type": "string", "default": "outputs"},
            "level": {"type": "integer", "default": 1, "required": False, "allowed": [1, 2]},
        },
    },
}

COMPLETE_CONFIG_DIFF_ANALYSIS = {
    "inputs": {
        "reference_elev": {
            "path_to_elev": "to_complete_with_an_elev",
            "nodata": "to_complete_with_user_nodata",
            "from_vcrs": "to_complete_with_vcrs",
            "to_vcrs": "to_complete_with_vcrs",
        },
        "to_be_aligned_elev": {
            "path_to_elev": "to_complete_with_an_elev",
            "nodata": "to_complete_with_user_nodata",
            "from_vcrs": "to_complete_with_vcrs",
            "to_vcrs": "to_complete_with_vcrs",
            "path_to_mask": "to_complete_with_a_mask",
        },
    },
    "outputs": {
        "level": "to_complete_with_level_1_or_2",
        "path": "to_complete_with_a_folder_name",
        "output_grid": "to_complete_with_output_grid_value",
    },
    "coregistration": {
        "sampling_grid": "to_complete_with_sampling_grid_value",
        "step_one": {
            "method": "to_complete_with_coreg_method",
            "extra_informations": "to_complete_with_dict_of_parameters",
        },
        "step_two": {
            "method": "to_complete_with_coreg_method",
            "extra_informations": "to_complete_with_dict_of_parameters",
        },
        "step_three": {
            "method": "to_complete_with_coreg_method",
            "extra_informations": "to_complete_with_dict_of_parameters",
        },
    },
    "statistics": "to_complete_with_list_of_statistics",
}

COMPLETE_CONFIG_TOPO_SUMMARY = {
    "inputs": {
        "reference_elev": {
            "path_to_elev": "to_complete_with_an_elev",
            "nodata": "to_complete_with_user_nodata",
            "from_vcrs": "to_complete_with_vcrs",
            "to_vcrs": "to_complete_with_vcrs",
            "path_to_mask": "to_complete_with_a_mask",
        },
    },
    "outputs": {"level": "to_complete_with_level_1_or_2", "path": "to_complete_with_a_folder_name"},
    "statistics": "to_complete_with_list_of_statistics",
    "terrain_attributes": "to_complete_with_a_list_OR_dict_with_extra_information",
}

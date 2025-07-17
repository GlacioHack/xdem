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

import os
from typing import Any, Dict

from cerberus import Validator
from pyproj import CRS


class CustomValidator(Validator):  # type: ignore
    def _validate_path_exists(self, path_exists: bool, field: str, value: str) -> bool:
        """
        {'type': 'boolean'}
        """
        if path_exists and not os.path.exists(value):
            self._error(field, f"Path does not exist: {value}")
        return True

    def _validate_crs_dict(self, crs_dict: dict[str, Any], field: str, value: dict[str, Any]) -> bool:
        """
        {'type': 'boolean'}
        """
        if crs_dict:

            if not isinstance(value, dict):
                self._error(field, "CRS must be a dictionary")
                return False

            if len(value) != 1:
                self._error(field, "Only one of 'common', 'proj_grid', or 'epsg_code' must be defined")
                return False

            key = next(iter(value))
            val = value[key]

            if key == "common":
                if val not in ["Ellipsoid", "EGM08", "EGM96"]:
                    self._error(field, f"Invalid 'common' value: {val}")

            elif key == "proj_grid":
                if not isinstance(val, str):
                    self._error(field, "'proj_grid' must be a string path")
                elif not val.endswith(".tif"):
                    self._error(field, f"'proj_grid' must point to a .tif file: {val}")

            elif key == "epsg_code":
                if not isinstance(val, int):
                    self._error(field, "'epsg_code' must be an integer")
                else:
                    try:
                        _ = CRS.from_epsg(val)
                    except Exception:
                        self._error(field, f"Invalid EPSG code: {val}")
            else:
                self._error(field, f"Unknown keys in CRS: {key}")
                return False

            return True
        else:
            return False


INPUTS_DEM = {
    "dem": {"type": "string", "required": True, "path_exists": True},
    "nodata": {"type": ["integer", "float"], "required": False},
    "mask": {"type": "string", "required": False, "path_exists": True},
    "from_vcrs": {"type": "dict", "required": False, "crs_dict": True, "default": {"common": "EGM96"}},
    "to_vcrs": {"type": "dict", "required": False, "crs_dict": True, "default": {"common": "EGM96"}},
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

TERRAIN_ATTRIBUTES_DEFAULT = ["hillshade", "slope", "aspect", "curvature", "terrain_ruggedness_index", "rugosity"]

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

    if "terrain_attributes" not in validator.document:
        validator.document["terrain_attributes"] = TERRAIN_ATTRIBUTES_DEFAULT

    return validator.document


COMPARE_SCHEMA = {
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
        "default": {"path": "outputs", "level": 1},
        "schema": {
            "path": {"type": "string", "required": True, "default": "outputs"},
            "level": {"type": "integer", "default": 1, "required": False, "allowed": [1, 2]},
        },
    },
    "coregistration": {
        "type": "dict",
        "required": False,
        "default": {"step_one": {"method": "NuthKaab"}, "sampling_source": "reference_elev"},
        "schema": {
            "step_one": make_coreg_step(default_method="NuthKaab"),
            "step_two": make_coreg_step(required=False),
            "step_three": make_coreg_step(required=False),
            "sampling_source": {
                "type": "string",
                "allowed": ["reference_elev", "to_be_aligned_elev"],
                "default": "reference_elev",
                "required": False,
            },
        },
    },
    "statistics": {"type": "list", "required": False, "allowed": STATS_METHODS, "nullable": True},
}

INFO_SCHEMA = {
    "inputs": {
        "type": "dict",
        "schema": INPUTS_DEM,
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

VOLCHANGE_SCHEMA = {
    "inputs": {
        "type": "dict",
        "schema": INPUTS_DEM,
    },
    "outputs": {
        "type": "dict",
        "required": False,
        "default": {"path": "outputs", "dem": False},
        "schema": {
            "path": {"type": "string", "default": "outputs"},
            "dem": {"type": "boolean", "default": False},
        },
    },
}

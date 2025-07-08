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
Workflow class
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore
from cerberus import Validator
from geoutils import Mask
from geoutils.raster import RasterType

import xdem
from xdem import DEM
from xdem.coreg.base import InputCoregDict, OutputCoregDict


class Workflows(ABC):
    """
    Abstract Class for workflows
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the workflows class
        :param config_path: path to config file
        :return: None
        """

        self.config_path = config_path
        self.config = self.load_config()
        self.validate_yaml()

        self.outputs_folder = Path(self.config["outputs"]["path"])
        self.outputs_folder.mkdir(parents=True, exist_ok=True)

        self.path_png = self.outputs_folder / "png"
        self.path_png.mkdir(parents=True, exist_ok=True)

        self.save_dem = self.config["outputs"]["dem"]
        if self.save_dem:
            self.path_tiff = self.outputs_folder / "raw_DEM"
            self.path_tiff.mkdir(parents=True, exist_ok=True)

        self.dico_to_show = [
            ("Information about inputs", self.config["inputs"]),
        ]

    def load_config(self) -> Dict[str, Any]:
        """
        Load a configuration file
        :return: Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"File not found : {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def validate_yaml(self) -> None:
        """
        Validate the YAML file
        """
        v = Validator(self.schema)
        if not v.validate(self.config):
            for field, errors in v.errors.items():
                raise ValueError(f"User configuration mistakes in '{field}': {errors}")

    def generate_graph(self, dem: RasterType, title: str, **kwargs: Any) -> None:
        """
        Generate plot from a DEM
        :param dem: Digital Elevation model
        :param title: title of graph
        :return: None
        """

        plot_title = title.replace("_", " ")
        dem.plot(title=plot_title, **kwargs)
        plt.savefig(self.path_png / f"{title}.png")
        plt.close()

    def floats_process(
        self, dict_with_floats: Dict[str, Any] | InputCoregDict | OutputCoregDict | Any
    ) -> Dict[str, Any]:  # type: ignore
        """
        Allows rounding all floats present in a dictionary to two decimal places
        :param dict_with_floats: Dictionary with float
        :return: Dictionary with floats
        """
        if isinstance(dict_with_floats, dict):
            return {k: self.floats_process(v) for k, v in dict_with_floats.items()}
        elif isinstance(dict_with_floats, list):
            return [self.floats_process(elem) for elem in dict_with_floats]  # type: ignore
        elif isinstance(dict_with_floats, tuple):
            return tuple(self.floats_process(elem) for elem in dict_with_floats)  # type: ignore
        elif isinstance(dict_with_floats, (float, np.floating)):
            return round(float(dict_with_floats), 2)  # type: ignore
        else:
            return dict_with_floats

    @staticmethod
    def generate_dem(config_dem: Dict[str, Any]) -> tuple[DEM, Mask | None]:
        """
        Generate DEM from user configuration dictionary
        :param config_dem: Configuration dictionary
        :return: DEM
        """
        dem = xdem.DEM(config_dem["dem"])
        inlier_mask = None
        if "nodata" in config_dem:
            dem.set_nodata(config_dem["nodata"])
        if "mask" in config_dem:
            mask = gu.Vector(config_dem["mask"])
            inlier_mask = ~mask.create_mask(dem)

        return dem, inlier_mask

    @abstractmethod
    def create_html(self, list_dict: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Create HTML page from png files and table
        :param list_dict: list containing tuples of title and various dictionaries
        :return: None
        """

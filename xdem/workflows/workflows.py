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

import csv
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore
from geoutils import Raster
from geoutils.raster import RasterType
from yaml.dumper import SafeDumper  # type: ignore

import xdem
from xdem import DEM
from xdem.coreg.base import InputCoregDict, OutputCoregDict
from xdem.workflows.schemas import validate_configuration


class Workflows(ABC):
    """
    Abstract Class for workflows
    """

    def __init__(self, user_config: str | Dict[str, Any]) -> None:
        """
        Initialize the workflows class
        :param user_config: str path to a config file or dict as config
        :return: None
        """

        # Load configuration
        if isinstance(user_config, str):
            if not os.path.isfile(user_config):
                raise FileNotFoundError(f"{user_config} does not exist")
            self.config_path = user_config
            config_not_verify = self.load_config()
        elif isinstance(user_config, dict):
            config_not_verify = user_config
        else:
            raise ValueError(
                "The configuration should be provided either as a path to the configuration file"
                " or as a dictionary containing the configuration details."
            )

        self.config = validate_configuration(config_not_verify, self.schema)
        self.level = self.config["outputs"]["level"]

        self.outputs_folder = Path(self.config["outputs"]["path"])
        logging.info(f"Outputs folder: {self.outputs_folder.absolute()}")
        self.outputs_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Outputs will be saved at {self.outputs_folder}")

        for folder in ["plots", "rasters", "tables"]:
            Path(self.outputs_folder / folder).mkdir(parents=True, exist_ok=True)

        self.dico_to_show = [
            ("Information about inputs", self.config["inputs"]),
        ]

    class NoAliasDumper(SafeDumper):  # type: ignore
        """
        NoAliasDumper to avoid id in YAML file
        """

        def ignore_aliases(self, data: Any) -> bool:
            """
            avoid id in YAML file
            """
            return True

    def load_config(self) -> Dict[str, Any]:
        """
        Load a configuration file
        :return: Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"File not found : {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def generate_plot(self, dem: RasterType, title: str, mask_path: str = None, **kwargs: Any) -> None:
        """
        Generate plot from a DEM
        :param dem: Digital Elevation model
        :param title: title of graph
        :param mask_path: Path to mask
        :return: None
        """

        plot_title = title.replace("_", " ")

        if mask_path is None:
            dem.plot(title=plot_title, **kwargs)
            plt.savefig(self.outputs_folder / "plots" / f"{title}.png")
            plt.close()
        else:
            mask = gu.Vector(mask_path)
            mask = mask.crop(dem)
            dem.plot(title=plot_title, **kwargs)
            mask.plot(dem, ec="k", fc="none")
            plt.savefig(self.outputs_folder / "plots" / f"{title}.png")
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
    def load_dem(config_dem: Dict[str, Any] | None) -> tuple[DEM, Raster, str | None]:
        """
        Generate DEM from user configuration dictionary
        :param config_dem: Configuration dictionary
        :return: DEM
        """
        mask_path = None
        if config_dem is not None:
            dem = xdem.DEM(config_dem["path_to_elev"], downsample=config_dem["downsample"])
            inlier_mask = None
            from_vcrs = config_dem["from_vcrs"]
            to_vcrs = config_dem["to_vcrs"]
            if from_vcrs:
                dem.set_vcrs(from_vcrs)
            if to_vcrs:
                if from_vcrs != to_vcrs:
                    dem.to_vcrs(to_vcrs)
            if config_dem.get("force_source_nodata") is not None:
                dem.set_nodata(config_dem["force_source_nodata"], update_array=False, update_mask=False)
            if config_dem.get("path_to_mask") is not None:
                mask_path = config_dem["path_to_mask"]
                mask = gu.Vector(mask_path)
                inlier_mask = ~mask.create_mask(dem)

            return dem, inlier_mask, mask_path
        else:
            logging.warning("No DEM provided")
            return None, None, None  # type: ignore

    def remove_none(self, dico: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """
        Recursively remove all keys whose values are None from a dictionary,
        except for the key 'statistics'.
        :param dico: dictionary to clean
        :return: cleaned dictionary
        """
        if isinstance(dico, dict):
            cleaned_dict = {}
            for k, v in dico.items():

                if k == "statistics":
                    cleaned_dict[k] = v
                    continue

                cleaned_value = self.remove_none(v) if v is not None else None
                if cleaned_value is not None:
                    cleaned_dict[k] = cleaned_value

            return cleaned_dict

        elif isinstance(dico, list):
            cleaned_list = [self.remove_none(v) for v in dico if v is not None]
            return [v for v in cleaned_list if v is not None]

        else:
            return dico

    @abstractmethod
    def create_html(self, list_dict: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Create HTML page from png files and table
        :param list_dict: list containing tuples of title and various dictionaries
        :return: None
        """

    def save_stat_as_csv(self, data: dict[str, float], file_name: str) -> None:
        """
        Save the statistics into a CSV file
        :param data: Statistics dictionary
        :param file_name: Name of csv file
        """
        cleaned_data = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in data.items()}

        fieldnames = list(cleaned_data.keys())
        filename = self.outputs_folder / "tables" / f"{file_name}_stats.csv"

        with filename.open(mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(cleaned_data)

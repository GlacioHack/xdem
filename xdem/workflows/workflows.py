# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
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
import math
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import geoutils as gu
import numpy as np
import pyogrio.errors
import rasterio.errors
from geoutils import Raster
from geoutils.raster import RasterType
from pyproj import CRS
from rasterio.warp import calculate_default_transform

import xdem
from xdem import DEM
from xdem._misc import import_optional
from xdem.coreg.base import InputCoregDict, OutputCoregDict
from xdem.examples import _FILEPATHS_ALL
from xdem.vcrs import vertical_unit_symbol
from xdem.workflows.schemas import validate_configuration

# Inheritance of optional dependencies
try:
    from yaml.dumper import SafeDumper  # type: ignore

    _HAS_YAML = True
except ImportError:
    SafeDumper = object
    _HAS_YAML = False

try:
    import plutoprint

    _HAS_PLUTOPRINT = True
except ImportError:
    _HAS_PLUTOPRINT = False


_ALIAS = {
    "mean": "Mean",
    "median": "Median",
    "max": "Maximum",
    "min": "Minimum",
    "sum": "Sum",
    "sumofsquares": "Sum of squares",
    "90thpercentile": "90th percentile",
    "le90": "LE90",
    "nmad": "NMAD",
    "rmse": "RMSE",
    "std": "Standard deviation",
    "standarddeviation": "Standard deviation",
    "validcount": "Valid count",
    "totalcount": "Total count",
    "percentagevalidpoints": "Percentage valid points",
}


class Workflows(ABC):
    """
    Abstract Class for workflows
    """

    def __init__(self, user_config: str | Dict[str, Any], output: str | None = None) -> None:
        """
        Initialize the workflows class

        :param user_config: Path to a config file or dict as config.

        :return: None
        """

        mpl = import_optional("matplotlib")

        # Default parameters for plots
        mpl.rcParams["font.size"] = "10"

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

        # If output folder was forced by user CLI argument, override the one in the config file
        if output is not None:
            self.outputs_folder = Path(output)
        else:
            self.outputs_folder = Path(self.config["outputs"]["path"])

        logging.info(f"Outputs folder: {self.outputs_folder.absolute()}")
        self.outputs_folder.mkdir(parents=True, exist_ok=True)

    def create_output_dir(self, sub_dir: Path | None = None) -> None:
        """
        Create sub directories table/rasters/tables

        :param sub_dir: path to replace outputs_folder
        :return: None
        """
        if sub_dir:
            self.outputs_folder = sub_dir
            self.outputs_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Outputs will be saved at {self.outputs_folder}")

        for folder in ["plots", "rasters", "tables"]:
            Path(self.outputs_folder / folder).mkdir(parents=True, exist_ok=True)

        self.dask = None
        self.mp_config = None
        if self.config.get("scalability", None) is not None:
            sca = self.config["scalability"]
            self.dask = sca.get("dask", None)
            if self.dask is None:
                from geoutils.multiproc import MultiprocConfig

                chunk_size = sca["multiprocess"]["chunk_size"]
                nb_workers = sca["multiprocess"].get("nb_workers", None)
                from geoutils.raster import ClusterGenerator

                cluster = None
                if nb_workers:
                    cluster = ClusterGenerator("multi", nb_workers)
                self.mp_config = MultiprocConfig(chunk_size=chunk_size, cluster=cluster)
            else:
                self.mp_config = None
                self.dask["chunks"]["band"] = 1

    class NoAliasDumper(SafeDumper):  # type: ignore
        """
        NoAliasDumper to avoid id in YAML file
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if not _HAS_YAML:
                import_optional("yaml", package_name="pyyaml")
            super().__init__(*args, **kwargs)

        def ignore_aliases(self, data: Any) -> bool:
            """
            Avoid id in YAML file
            """
            return True

    def load_config(self) -> Dict[str, Any]:
        """
        Load a configuration file
        Note: all null values in the .yaml are translated to None in the dict
        :return: Configuration dictionary
        """
        yaml = import_optional("yaml", package_name="pyyaml")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"File not found : {self.config_path}")

        if Path(self.config_path).suffix not in [".yaml", ".yml"]:
            raise ValueError("Unsupported configuration file format. Please use .yaml, or .yml file.")

        with open(self.config_path) as f:
            config_dict = yaml.safe_load(f)

            def replace_none_str_with_none_type(some_dict: Dict[str, Any]) -> Dict[str, Any]:
                """Replace all "None" (None after serialization) values to None"""
                for k, v in some_dict.items():
                    if isinstance(v, dict):
                        some_dict[k] = replace_none_str_with_none_type(v)
                    elif v == "None":
                        some_dict[k] = None
                    else:
                        some_dict[k] = v
                return some_dict

            return replace_none_str_with_none_type(config_dict)

    @staticmethod
    def load_dem(
        config_dem: Dict[str, Any], dask: Dict[str, Any] | None = None
    ) -> tuple[RasterType, RasterType | None, str | None]:
        """
        Generate DEM from user configuration dictionary.

        :param config_dem: Configuration dictionary.

        :return: DEM.
        """

        logging.info("Read data with Xarray" if dask is not None else "Read data without Xarray")
        path_to_elev = config_dem["path_to_elev"]

        # If alias, get its path
        if path_to_elev in list(_FILEPATHS_ALL.keys()):
            path_to_elev = xdem.examples.get_path(path_to_elev)

        # Get default value
        config_dem["downsample"] = config_dem.get("downsample", 1)
        from xdem import open_dem

        force_vcrs = config_dem.get("force_vcrs", None)

        if dask is not None:
            chunks = dask["chunks"]
            dem = open_dem(path_to_elev, chunks=chunks, downsample=config_dem.get("downsample", 1))
            if force_vcrs:
                dem.dem.set_vcrs(force_vcrs)
            if config_dem.get("force_source_nodata") is not None:
                dem.dem.set_nodata(config_dem["force_source_nodata"], update_array=False, update_mask=False)

        else:
            dem = DEM(path_to_elev)
            if force_vcrs:
                dem.set_vcrs(force_vcrs)
            if config_dem.get("force_source_nodata") is not None:
                dem.set_nodata(config_dem["force_source_nodata"], update_array=False, update_mask=False)

        inlier_mask = None
        mask_path = None

        if config_dem.get("path_to_mask") is not None:
            mask_path = config_dem["path_to_mask"]

            # If alias, get its path
            if mask_path in list(_FILEPATHS_ALL.keys()):
                mask_path = xdem.examples.get_path(mask_path)

            # Treat the mask according to its type (Vector or Raster)
            try:
                if dask is not None:
                    mask = gu.open_vector(mask_path, chunks=1)
                    inlier_mask = ~mask.vct.create_mask(dem.dem)
                else:
                    mask = gu.Vector(mask_path)
                    inlier_mask = ~mask.create_mask(dem)

            except pyogrio.errors.DataSourceError as vector_error:
                try:
                    if dask is not None:
                        chunks = dask["chunks"]
                        inlier_mask = gu.open_raster(
                            mask_path, chunks=chunks, downsample=config_dem.get("downsample", 1)
                        )
                        inlier_mask.rst.data = inlier_mask.rst.data.astype(bool)
                        inlier_mask.rst.data = inlier_mask.rst.data.filled(False)
                        inlier_mask = inlier_mask.rst.reproject(dem, silent=True)
                    else:
                        inlier_mask = Raster(mask_path).astype(bool)
                        inlier_mask.data = inlier_mask.data.filled(False)
                        inlier_mask = inlier_mask.reproject(dem, silent=True)

                except rasterio.errors.RasterioIOError as raster_error:
                    raise ValueError(
                        f"You provided a 'path_to_mask' value that is not recognised as a mask. "
                        f"Vector error: {vector_error}. "
                        f"Raster error: {raster_error}."
                    ) from raster_error

        return dem, inlier_mask, mask_path

    def remove_none(self, dico: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """
        Recursively remove all keys whose values are None from a dictionary, except for the key 'statistics'.

        :param dico: Dictionary to clean.

        :return: Cleaned dictionary.
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
        Create HTML page from png files and table.

        :param list_dict: List containing tuples of title and various dictionaries.
        :return: None
        """

    def generate_pdf(self) -> None:
        """
        Create PDF report page from HTML page.

        :return: None
        """
        if self.config["outputs"]["generate_pdf"] and _HAS_PLUTOPRINT:
            book = plutoprint.Book(plutoprint.PAGE_SIZE_A4, plutoprint.PAGE_MARGINS_NARROW)
            book.load_url((self.outputs_folder / "report.html").resolve().as_uri())

            # Export the entire document to PDF
            book.write_to_pdf(str(self.outputs_folder / "report.pdf"))
            logging.info("Report generated in " + str(self.outputs_folder / "report.pdf"))

    def save_stat_as_csv(self, data: dict[str, float], file_name: str) -> None:
        """
        Save the statistics into a CSV file.

        :param data: Statistics dictionary.
        :param file_name: Name of csv file.
        """
        cleaned_data = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in data.items()}

        fieldnames = list(cleaned_data.keys())
        filename = self.outputs_folder / "tables" / f"{file_name}_stats.csv"

        with filename.open(mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(cleaned_data)

    def format_values_stats(self, key: str, val: Union[float, int]) -> str:
        """Format values for the statistics."""
        if "count" in key.lower():
            return str(int(val))
        if "percentage" in key.lower():
            return f"{val:.2f}" + "%"
        elif abs(val) > 10e4:
            return np.format_float_scientific(val, precision=3)
        elif abs(val) < 10e-4:
            return np.format_float_scientific(val, precision=3)
        else:
            return f"{val:.3f}"

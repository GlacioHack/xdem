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
Topo class from workflows.
"""

import matplotlib

matplotlib.use("Agg")

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from geoutils.raster import Raster

import xdem
from xdem._misc import import_optional
from xdem.workflows import toolbox
from xdem.workflows.schemas import TOPO_SCHEMA
from xdem.workflows.workflows import _ALIAS, Workflows


class Topo(Workflows):
    """
    Topo class from workflows.
    """

    def __init__(self, config_dem: str | Dict[str, Any], output: str | None = None) -> None:
        """
        Initialize Topo class
        :param config_dem: Path to a user configuration file
        """

        yaml = import_optional("yaml", package_name="pyyaml")

        self.elapsed: float | None = None
        self.schema = TOPO_SCHEMA

        super().__init__(config_dem, output)

        self.config_attributes = self.config["terrain_attributes"]
        if isinstance(self.config_attributes, dict):
            self.list_attributes = list(self.config_attributes.keys())
        else:
            self.list_attributes = self.config_attributes

        if isinstance(self.config["inputs"], dict):
            self.config["inputs"] = [self.config["inputs"]]

        yaml_str = yaml.dump(self.config, allow_unicode=True, Dumper=self.NoAliasDumper)
        Path(self.outputs_folder / "used_config.yaml").write_text(yaml_str, encoding="utf-8")

        self.config = self.remove_none(self.config)  # type: ignore

    def _load_data(self, input: Dict[str, Any]) -> None:
        """
        Load data defined in config file.

        :param input: input info (dict) to load
        :return: None
        """
        if "scalability" in self.config:
            dask = self.config["scalability"]["dask"]
        else:
            dask = None
        self.dem, self.inlier_mask, path_to_mask = self.load_dem(input, dask)

        toolbox.generate_plot(
            self.dem if isinstance(self.dem, Raster) else self.dem.dem,
            file_path=self.outputs_folder / "plots" / f"elev_map.png",
            title="Elevation",
        )

        if self.inlier_mask is not None:
            if isinstance(self.inlier_mask, Raster):
                inlier_mask = self.inlier_mask.reproject(self.dem, silent=True)
                inlier_mask = inlier_mask.crop(self.dem)
            else:
                inlier_mask = self.inlier_mask.rst.reproject(self.dem.dem, silent=True)
                inlier_mask = inlier_mask.rst.crop(self.dem.dem)

            if isinstance(self.dem, Raster):
                self.dem.set_mask(~inlier_mask)
            else:
                self.dem.dem.data = np.where(inlier_mask.rst.data, self.dem.dem.data, np.nan)

            toolbox.generate_plot(
                self.dem if isinstance(self.dem, Raster) else self.dem.dem,
                title="Masked elevation",
                file_path=self.outputs_folder / "plots" / f"masked_elev_map.png",
            )

    def run(self) -> None:
        """
        Run function for the topography workflow.
        :return: None
        """

        t0 = time.time()
        self.dico_to_show = []
        general_output = self.outputs_folder

        # For each input
        for k, input in enumerate(self.config["inputs"]):
            self.dico_to_show.append(
                [
                    ("Information about inputs", input),
                ]
            )

            # If several inputs, corresponding outputs stored in outputs_folder/dem_index
            if len(self.config["inputs"]) > 1:
                self.outputs_folder = general_output / ("dem_" + str(k))
                logging.info(f"Elevation input #{k}")

            self.create_output_dir()
            self._load_data(input)

            # Global information
            dem_informations = toolbox.return_info(self.dem if isinstance(self.dem, Raster) else self.dem.dem)
            self.dico_to_show[k].append(("Elevation information", dem_informations))

            # Statistics
            list_metrics = self.config["statistics"]
            if list_metrics is not None:
                if isinstance(self.dem, Raster):
                    stats_dem = self.dem.get_stats(list_metrics)
                else:
                    stats_dem = self.dem.dem.get_stats(list_metrics)

                stats_dem = {_ALIAS.get(k, k): v for k, v in stats_dem.items()}
                self.save_stat_as_csv(stats_dem, "stats_elev")
                self.dico_to_show[k].append(("Statistics", toolbox.floats_process(stats_dem)))
                logging.info(f"Computing metrics on elevation: {list_metrics}")

            # Terrain attributes
            if self.list_attributes is not None and len(self.list_attributes):
                toolbox.generate_terrain_attributes(
                    self.dem if isinstance(self.dem, Raster) else self.dem.dem,
                    self.list_attributes,
                    self.outputs_folder,
                    self.config_attributes,
                    (
                        self.config.get("reproject", None) is None
                        or self.config["reproject"].get("crs", None) is None
                        or self.config["reproject"]["crs"] is True
                    ),
                    self.config["reproject"]["crs"]
                    if "reproject" in self.config and "crs" in self.config["reproject"]
                    else None,
                    self.level > 1,
                )
            else:
                logging.info("No terrain attributes to compute")

            t1 = time.time()
            self.elapsed = t1 - t0

            self.create_html(self.dico_to_show[k])
            self.generate_pdf()

            # Remove empty folder
            for folder in self.outputs_folder.rglob("*"):
                if folder.is_dir():
                    try:
                        folder.rmdir()
                    except OSError:
                        pass

    def create_html(self, list_dict: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Create HTML page from png files and table
        :param list_dict: list containing tuples of title and various dictionaries
        :return: None
        """

        html = "<html>\n<head><meta charset='UTF-8'><title>Topographic summary results</title></head>\n<body>\n"

        # Title and version/date/time summary
        html += "<h1>Topography summary report — xDEM</h1>\n"

        html += f"<p>xDEM version: {xdem.__version__}</p>"
        html += f"<p>Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>"
        html += f"<p>Computing time: {self.elapsed:.2f} seconds</p>"

        html += "<h2>Elevation input</h2>\n"
        html += "<img src='plots/elev_map.png' alt='Image PNG' style='width: 100%; height: auto;'>\n"

        if self.inlier_mask is not None:
            html += "<h2>Masked elevation data</h2>\n"
            html += "<img src='plots/masked_elev_map.png' alt='Image PNG' style='width: 100%; height: auto;'>\n"

        for title, dictionary in list_dict:
            html += "<div style='clear: both; margin-bottom: 30px;'>\n"  # type: ignore
            html += f"<h2>{title}</h2>\n"
            html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            html += "<tr><th>Information</th><th>Value</th></tr>\n"
            for key, val in dictionary.items():
                if "statistics" in title.lower():
                    html += f"<tr><td>{key}</td><td>{self.format_values_stats(key, val)}</td></tr>\n"
                else:
                    html += f"<tr><td>{key}</td><td>{val}</td></tr>\n"
            html += "</table>\n"
            html += "</div>\n"

        # Terrain attributes
        if self.list_attributes is not None:
            html += "<h2>Terrain attributes</h2>\n"
            html += "<img src='plots/terrain_attributes_map.png' alt='Image PNG' style='width: 100%; height: auto;'>\n"

        html += "</body>\n</html>"

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)

        logging.info("Report generated in " + str(self.outputs_folder / "report.html"))

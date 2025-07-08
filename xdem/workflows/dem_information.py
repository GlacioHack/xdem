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
DemInformation class from workflows.
"""

import math
import os
from typing import Any

import matplotlib.pyplot as plt

import xdem
from xdem.workflows.workflows import Workflows


class DemInformation(Workflows):
    """
    DemInformation class from workflows.
    """

    def __init__(self, config_dem: str):
        """
        Initialize DemInformation class
        :param config_dem: Path to a user configuration file
        """

        self.schema = {
            "inputs": {
                "type": "dict",
                "schema": {
                    "dem": {"type": "string", "required": True},
                    "nodata": {"type": ["integer", "float"], "required": False},
                    "mask": {"type": "string", "required": False},
                },
            },
            "statistics": {"type": "list", "required": False},
            "terrain_attributes": {
                "type": "list",
                "required": False,
                "schema": {
                    "type": "string",
                    "allowed": [
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
                    ],
                },
            },
            "outputs": {
                "type": "dict",
                "schema": {
                    "path": {"type": "string"},
                    "terrain_attributes": {"type": "boolean"},
                    "dem": {"type": "boolean"},
                },
            },
        }

        super().__init__(config_dem)

        # Verify entry
        assert os.path.isfile(self.config["inputs"]["dem"]), f"{self.config['inputs']['dem']} does not exist"
        self.dem, self.inlier_mask = self.generate_dem(self.config["inputs"])
        self.generate_graph(self.dem, "Digitial_elevation_model")
        self.save_terrain_attributes = self.config["outputs"]["terrain_attributes"]

        if self.save_terrain_attributes:
            self.path_terrain = self.outputs_folder / "terrain_attributes"
            self.path_terrain.mkdir(parents=True, exist_ok=True)

    def generate_terrain_attributes(self) -> None:
        """
        Generates an image png containing the plots of the terrain attributes requested by the user.
        :return: None
        """

        if "terrain_attributes" not in self.config:
            list_attributes = ["hillshade", "slope", "aspect", "curvature", "terrain_ruggedness_index", "rugosity"]
        else:
            list_attributes = self.config["terrain_attributes"]

        attributes = xdem.terrain.get_terrain_attribute(
            self.dem.data,
            resolution=self.dem.res,
            attribute=list_attributes,
        )

        if self.save_terrain_attributes:
            from_str_to_fun = {
                "slope": lambda: self.dem.slope(),
                "aspect": lambda: self.dem.aspect(),
                "hillshade": lambda: self.dem.hillshade(),
                "curvature": lambda: self.dem.curvature(),
                "planform_curvature": lambda: self.dem.planform_curvature(),
                "profile_curvature": lambda: self.dem.profile_curvature(),
                "maximum_curvature": lambda: self.dem.maximum_curvature(),
                "topographic_position_index": lambda: self.dem.topographic_position_index(),
                "terrain_ruggedness_index": lambda: self.dem.terrain_ruggedness_index(),
                "roughness": lambda: self.dem.roughness(),
                "rugosity": lambda: self.dem.rugosity(),
                "fractal_roughness": lambda: self.dem.fractal_roughness(),
            }
            for attr in list_attributes:
                attribute = from_str_to_fun[attr]()
                attribute.save(self.path_terrain / f"{attr}.tif")

        n = len(attributes)

        ncols = 2
        nrows = math.ceil(n / ncols)

        plt.figure(figsize=(8, 6.5))

        plt_extent = [self.dem.bounds.left, self.dem.bounds.right, self.dem.bounds.bottom, self.dem.bounds.top]

        cmaps = [
            "Greys_r",
            "Reds",
            "twilight",
            "RdGy_r",
            "RdGy_r",
            "RdGy_r",
            "Purples",
            "YlOrRd",
            "Spectral",
            "Oranges",
            "Reds",
        ]
        labels = [
            "Hillshade",
            "Slope (°)",
            "Aspect (°)",
            "Curvature (100 / m)",
            "Planform curvature (100 / m)",
            "Profile curvature (100 / m)",
            "Terrain Ruggedness Index",
            "Rugosity",
            "Topographic position index (m)",
            "Roughness (m)",
            "Fractal roughness (dimensions)",
        ]
        vlims = [(None, None)] * n
        if n > 3:
            vlims[3] = [-2, 2]  # type: ignore

        for i in range(n):
            plt.subplot(nrows, ncols, i + 1)

            cmap = cmaps[i] if i < len(cmaps) else "viridis"
            label = labels[i] if i < len(labels) else f"Attribute {i + 1}"
            vmin, vmax = vlims[i] if i < len(vlims) else (None, None)

            plt.imshow(attributes[i].squeeze(), cmap=cmap, extent=plt_extent, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            cbar.set_label(label)
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.savefig(self.path_png / "terrain_attributes.png")
        plt.close()

    def run(self) -> None:
        """
        Run function for the coregistration workflow
        :return: None
        """

        # Global information
        dem_informations = {
            "Driver": self.dem.driver,
            "Filename": self.dem.filename,
            "Grid size": self.dem.vcrs_grid,
            "Number of band": self.dem.bands,
            "Data types": self.dem.dtype,
            "Nodata Value": self.dem.nodata,
            "Pixel interpretation": self.dem.area_or_point,
            "Pixel size": self.dem.res,
            "Width": self.dem.width,
            "Height": self.dem.height,
            "Transform": self.dem.transform,
        }

        # Statistics
        if "statistics" not in self.config:
            stats_dem = self.dem.get_stats()
            stats_dem_mask = self.dem.get_stats(inlier_mask=self.inlier_mask)
        else:
            stats_dem = self.dem.get_stats(self.config["statistics"])
            stats_dem_mask = self.dem.get_stats(self.config["statistics"], inlier_mask=self.inlier_mask)

        # Terrain attributes
        self.generate_terrain_attributes()

        # Generate HTML
        self.dico_to_show.append(("DEM information", dem_informations))
        self.dico_to_show.append(("Global statistics", self.floats_process(stats_dem)))
        self.dico_to_show.append(("Mask statistics", self.floats_process(stats_dem_mask)))

        self.create_html(self.dico_to_show)

    def create_html(self, list_dict: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Create html page from png files and table
        :param list_dict: list containing tuples of title and various dictionaries
        :return: None
        """

        html = "<html>\n<head><meta charset='UTF-8'><title>Qualify DEM results</title></head>\n<body>\n"

        html += "<h2>Digital Elevation Model</h2>\n"
        html += "<img src='png/Digitial_elevation_model.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        for title, dictionary in list_dict:
            html += "<div style='clear: both; margin-bottom: 30px;'>\n"  # type: ignore
            html += f"<h2>{title}</h2>\n"
            html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            html += "<tr><th>Information</th><th>Value</th></tr>\n"
            for cle, valeur in dictionary.items():
                html += f"<tr><td>{cle}</td><td>{valeur}</td></tr>\n"
            html += "</table>\n"
            html += "</div>\n"

        html += "<h2>Terrain attributes</h2>\n"
        html += "<img src='png/terrain_attributes.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        html += "</body>\n</html>"

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)

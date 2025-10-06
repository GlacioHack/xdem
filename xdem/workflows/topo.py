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
Topo class from workflows.
"""
import logging
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import yaml  # type: ignore

import xdem
from xdem.workflows.schemas import TOPO_SCHEMA
from xdem.workflows.workflows import Workflows


class Topo(Workflows):
    """
    Topo class from workflows.
    """

    def __init__(self, config_dem: str | Dict[str, Any]):
        """
        Initialize Topo class
        :param config_dem: Path to a user configuration file
        """

        self.schema = TOPO_SCHEMA

        super().__init__(config_dem)

        self.dem, self.inlier_mask, path_to_mask = self.load_dem(self.config["inputs"]["reference_elev"])
        self.generate_plot(self.dem, "elev_map", cmap="terrain", cbar_title="Elevation (m)")

        if self.inlier_mask is not None:
            self.generate_plot(
                self.dem,
                "masked_elevation",
                mask_path=path_to_mask,
                cmap="terrain",
                cbar_title="Elevation (m)",
            )

        self.config_attributes = self.config["terrain_attributes"]
        if isinstance(self.config_attributes, dict):
            self.list_attributes = list(self.config_attributes.keys())
        else:
            self.list_attributes = self.config_attributes

        yaml_str = yaml.dump(self.config, allow_unicode=True, Dumper=self.NoAliasDumper)
        Path(self.outputs_folder / "used_config.yaml").write_text(yaml_str, encoding="utf-8")

        self.config = self.remove_none(self.config)  # type: ignore

    def generate_terrain_attributes_tiff(self) -> None:
        """
        Generate terrain attributes tiff
        """

        attribute_extra = {}

        from_str_to_fun = {
            "slope": lambda: self.dem.slope(**attribute_extra),
            "aspect": lambda: self.dem.aspect(**attribute_extra),
            "hillshade": lambda: self.dem.hillshade(**attribute_extra),
            "curvature": lambda: self.dem.curvature(**attribute_extra),
            "planform_curvature": lambda: self.dem.planform_curvature(**attribute_extra),
            "profile_curvature": lambda: self.dem.profile_curvature(**attribute_extra),
            "maximum_curvature": lambda: self.dem.maximum_curvature(**attribute_extra),
            "topographic_position_index": lambda: self.dem.topographic_position_index(**attribute_extra),
            "terrain_ruggedness_index": lambda: self.dem.terrain_ruggedness_index(**attribute_extra),
            "roughness": lambda: self.dem.roughness(**attribute_extra),
            "rugosity": lambda: self.dem.rugosity(**attribute_extra),
            "fractal_roughness": lambda: self.dem.fractal_roughness(**attribute_extra),
        }
        for attr in self.list_attributes:
            if isinstance(self.config_attributes, dict):
                attribute_extra = self.config_attributes.get(attr).get("extra_information", {})  # type: ignore
            attribute = from_str_to_fun[attr]()
            logging.info(f"Saving {attr} as a raster file ({attr}.tif)")
            attribute.save(self.outputs_folder / "rasters" / f"{attr}.tif")

    def generate_terrain_attributes_png(self) -> None:
        """
        Generates an image png containing the plots of the terrain attributes requested by the user.
        :return: None
        """

        logging.info(f"Computing attributes : {self.list_attributes}")

        attributes = xdem.terrain.get_terrain_attribute(
            self.dem.data,
            resolution=self.dem.res,
            attribute=self.list_attributes,
        )

        n = len(attributes)

        ncols = 2
        nrows = math.ceil(n / ncols)

        plt.figure(figsize=(8, 6.5))

        plt_extent = [self.dem.bounds.left, self.dem.bounds.right, self.dem.bounds.bottom, self.dem.bounds.top]

        attribute_params = {
            "hillshade": {"label": "Hillshade", "cmap": "Greys_r", "vlim": (None, None)},
            "slope": {"label": "Slope (°)", "cmap": "Reds", "vlim": (None, None)},
            "aspect": {"label": "Aspect (°)", "cmap": "twilight", "vlim": (None, None)},
            "curvature": {"label": "Curvature (100 / m)", "cmap": "RdGy_r", "vlim": (-2, 2)},
            "planform_curvature": {"label": "Planform curvature (100 / m)", "cmap": "RdGy_r", "vlim": (-2, 2)},
            "profile_curvature": {"label": "Profile curvature (100 / m)", "cmap": "RdGy_r", "vlim": (-2, 2)},
            "terrain_ruggedness_index": {"label": "Terrain Ruggedness Index", "cmap": "Purples", "vlim": (None, None)},
            "rugosity": {"label": "Rugosity", "cmap": "YlOrRd", "vlim": (None, None)},
            "topographic_position_index": {
                "label": "Topographic position index (m)",
                "cmap": "Spectral",
                "vlim": (None, None),
            },
            "roughness": {"label": "Roughness (m)", "cmap": "Oranges", "vlim": (None, None)},
            "fractal_dimension": {"label": "Fractal roughness (dimensions)", "cmap": "Reds", "vlim": (None, None)},
        }

        for i, attr in enumerate(self.list_attributes):
            plt.subplot(nrows, ncols, i + 1)

            params = attribute_params.get(attr, {})
            cmap = params.get("cmap", "viridis")
            label = params.get("label", f"Attribute {i + 1}")
            vmin, vmax = params.get("vlim", (None, None))

            plt.imshow(attributes[i].squeeze(), cmap=cmap, extent=plt_extent, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            cbar.set_label(label)
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.savefig(self.outputs_folder / "plots" / "terrain_attributes_map.png")
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
            "Bounds": self.dem.bounds,
        }

        # Statistics
        list_metrics = self.config["statistics"]
        if list_metrics is not None:
            stats_dem = self.dem.get_stats(list_metrics)
            self.save_stat_as_csv(stats_dem, "stats_elev")
            stats_dem_mask = self.dem.get_stats(list_metrics, inlier_mask=self.inlier_mask)
            if self.inlier_mask is not None:
                self.save_stat_as_csv(stats_dem_mask, "stats_elev_mask")
            logging.info(f"Computing metrics on reference elevation: {list_metrics}")

        # Terrain attributes
        if self.list_attributes is not None:
            self.generate_terrain_attributes_png()
            if self.level > 1:
                self.generate_terrain_attributes_tiff()
        else:
            logging.info("Computing terrain attributes: None")

        # Generate HTML
        self.dico_to_show.append(("Elevation information", dem_informations))
        if list_metrics is not None:
            self.dico_to_show.append(("Global statistics", self.floats_process(stats_dem)))
        if self.inlier_mask is not None:
            self.dico_to_show.append(("Mask statistics", self.floats_process(stats_dem_mask)))

        self.create_html(self.dico_to_show)

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

        html += "<h2>Elevation Model</h2>\n"
        html += "<img src='plots/elev_map.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        if self.inlier_mask is not None:
            html += "<h2>Masked elevation Model</h2>\n"
            html += "<img src='plots/masked_elevation.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        for title, dictionary in list_dict:
            html += "<div style='clear: both; margin-bottom: 30px;'>\n"  # type: ignore
            html += f"<h2>{title}</h2>\n"
            html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            html += "<tr><th>Information</th><th>Value</th></tr>\n"
            for key, value in dictionary.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            html += "</table>\n"
            html += "</div>\n"

        html += "<h2>Terrain attributes</h2>\n"
        html += "<img src='plots/terrain_attributes_map.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        html += "</body>\n</html>"

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)

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
import glob
import logging
import math
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import yaml  # type: ignore
from geoutils.raster import RasterType, map_multiproc_collect
from geoutils.raster.distributed_computing import MultiprocConfig
from numpy import floating

import xdem
from xdem.workflows.schemas import TOPO_SCHEMA
from xdem.workflows.workflows import Workflows

# flake8: noqa: B023


class TopoBigData(Workflows):
    """ """

    def __init__(self, config_elev: str | Dict[str, Any]):
        """
        Initialize Topo class
        :param config_elev: Path to a user configuration file
        """

        self.schema = TOPO_SCHEMA

        super().__init__(config_elev)

        self.path_elev = self.config["inputs"]["reference_elev"]["path_to_elev"]
        self.elev, self.inlier_mask, path_to_mask = self.load_elev(self.config["inputs"]["reference_elev"])

        self.config_attributes = self.config["terrain_attributes"]
        if isinstance(self.config_attributes, dict):
            self.list_attributes = list(self.config_attributes.keys())
        else:
            self.list_attributes = self.config_attributes
        self.list_metrics = self.config["statistics"]

        self.nb_workers = self.config["big_data"]["nb_workers"]
        self.block_size = self.config["big_data"]["block_size"]
        from geoutils.raster.tiling import compute_tiling

        self.shape_tiling_grid = compute_tiling(self.block_size, self.elev.shape, self.elev.shape).shape
        self.mp_config = MultiprocConfig(chunk_size=self.block_size)

        yaml_str = yaml.dump(self.config, allow_unicode=True, Dumper=self.NoAliasDumper)
        Path(self.outputs_folder / "used_config.yaml").write_text(yaml_str, encoding="utf-8")

        self.config = self.remove_none(self.config)  # type: ignore

        print("toto")

    def generate_terrain_attributes_tiff(self) -> None:
        """
        Generate terrain attributes tiff
        """

        for attr in self.list_attributes:
            logging.info(f"Saving {attr} as a raster file ({attr}.tif)")
            self.mp_config.outfile = f"{self.outputs_folder}/rasters/{attr}.tif"
            if isinstance(self.config_attributes, dict):
                attribute_extra = self.config_attributes.get(attr, {}).get("extra_information", {})
            else:
                attribute_extra = {}

            from_str_to_fun = {
                "slope": lambda: self.elev.slope(mp_config=self.mp_config, **attribute_extra),
                "aspect": lambda: self.elev.aspect(mp_config=self.mp_config, **attribute_extra),
                "hillshade": lambda: self.elev.hillshade(mp_config=self.mp_config, **attribute_extra),
                "curvature": lambda: self.elev.curvature(mp_config=self.mp_config, **attribute_extra),
                "planform_curvature": lambda: self.elev.planform_curvature(mp_config=self.mp_config, **attribute_extra),
                "profile_curvature": lambda: self.elev.profile_curvature(mp_config=self.mp_config, **attribute_extra),
                "maximum_curvature": lambda: self.elev.maximum_curvature(mp_config=self.mp_config, **attribute_extra),
                "topographic_position_index": lambda: self.elev.topographic_position_index(
                    mp_config=self.mp_config, **attribute_extra
                ),
                "terrain_ruggedness_index": lambda: self.elev.terrain_ruggedness_index(
                    mp_config=self.mp_config, **attribute_extra
                ),
                "roughness": lambda: self.elev.roughness(mp_config=self.mp_config, **attribute_extra),
                "rugosity": lambda: self.elev.rugosity(mp_config=self.mp_config, **attribute_extra),
                "fractal_roughness": lambda: self.elev.fractal_roughness(mp_config=self.mp_config, **attribute_extra),
            }
            from_str_to_fun[attr]()

    @staticmethod
    def _wrapper_stats(raster: RasterType, metric: str, inlier_mask: RasterType) -> floating[Any]:
        """
        Wrapper function for raster statistics because of multiprocessing
        :param raster: Raster object (with block size)
        :param metric: metric to compute statistics
        :param inlier_mask: inlier mask
        """
        return raster.get_stats(metric, inlier_mask=inlier_mask)

    def run(self) -> None:
        """
        Run function for the coregistration workflow
        :return: None
        """

        # Global information
        elev_informations = {
            "Driver": self.elev.driver,
            "Filename": self.elev.filename,
            "Grid size": self.elev.vcrs_grid,
            "Number of band": self.elev.bands,
            "Data types": self.elev.dtype,
            "Nodata Value": self.elev.nodata,
            "Pixel interpretation": self.elev.area_or_point,
            "Pixel size": self.elev.res,
            "Width": self.elev.width,
            "Height": self.elev.height,
            "Transform": self.elev.transform,
            "Bounds": self.elev.bounds,
        }
        self.dico_to_show.append(("Elevation information", elev_informations))
        # Terrain attributes
        self.generate_terrain_attributes_tiff()

        # Statistics
        if self.list_metrics is not None:
            logging.info(f"Computing metrics on elevation: {self.list_metrics}")
            for metric in self.list_metrics:
                the_stats = map_multiproc_collect(
                    self._wrapper_stats, self.path_elev, self.mp_config, metric, self.inlier_mask
                )

                combinations = list(product(range(self.shape_tiling_grid[0]), range(self.shape_tiling_grid[1])))
                str_combinations = [f"{a}_{b}" for a, b in combinations]

                resultat = dict(zip(str_combinations, the_stats, strict=False))

                df = pd.DataFrame.from_dict(resultat, orient="index")
                df.to_csv(
                    self.outputs_folder / "tables" / f"{metric}.csv",
                )

                self.save_stats_tiles(self.outputs_folder / "plots" / f"{metric}.png", df, metric)

        self.create_html()
        #
        # Remove empty folder
        for folder in self.outputs_folder.rglob("*"):
            if folder.is_dir():
                try:
                    folder.rmdir()
                except OSError:
                    pass

    def create_html(self) -> None:
        """
        Create HTML page from png files and table
        :return: None
        """

        fichiers_png = glob.glob(os.path.join(self.outputs_folder, "**", "*.png"))
        noms_png = [os.path.basename(f) for f in fichiers_png]

        html = "<html><body><h1>Elevation Model</h1><div>"

        for title, dictionary in self.dico_to_show:
            html += "<div style='clear: both; margin-bottom: 30px;'>\n"
            html += f"<h2>{title}</h2>\n"
            html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            html += "<tr><th>Information</th><th>Value</th></tr>\n"
            for key, value in dictionary.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            html += "</table>\n"
            html += "</div>\n"

        for metric in noms_png:
            path = f"plots/{metric}"
            html += f'<img src="{path}" width="200">\n'
        html += "</div></body></html>"

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)


class Topo(Workflows):
    """
    Topo class from workflows.
    """

    def __init__(self, config_elev: str | Dict[str, Any]):
        """
        Initialize Topo class
        :param config_elev: Path to a user configuration file
        """

        self.schema = TOPO_SCHEMA

        super().__init__(config_elev)

        self.elev, self.inlier_mask, path_to_mask = self.load_elev(self.config["inputs"]["reference_elev"])
        self.generate_plot(self.elev, "elev_map", cmap="terrain", cbar_title="Elevation (m)")

        if self.inlier_mask is not None:
            self.generate_plot(
                self.elev,
                "masked_elev_map",
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
            "slope": lambda: self.elev.slope(**attribute_extra),
            "aspect": lambda: self.elev.aspect(**attribute_extra),
            "hillshade": lambda: self.elev.hillshade(**attribute_extra),
            "curvature": lambda: self.elev.curvature(**attribute_extra),
            "planform_curvature": lambda: self.elev.planform_curvature(**attribute_extra),
            "profile_curvature": lambda: self.elev.profile_curvature(**attribute_extra),
            "maximum_curvature": lambda: self.elev.maximum_curvature(**attribute_extra),
            "topographic_position_index": lambda: self.elev.topographic_position_index(**attribute_extra),
            "terrain_ruggedness_index": lambda: self.elev.terrain_ruggedness_index(**attribute_extra),
            "roughness": lambda: self.elev.roughness(**attribute_extra),
            "rugosity": lambda: self.elev.rugosity(**attribute_extra),
            "fractal_roughness": lambda: self.elev.fractal_roughness(**attribute_extra),
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
            self.elev.data,
            resolution=self.elev.res,
            attribute=self.list_attributes,
        )

        n = len(attributes)

        ncols = 2
        nrows = math.ceil(n / ncols)

        plt.figure(figsize=(8, 6.5))

        plt_extent = [self.elev.bounds.left, self.elev.bounds.right, self.elev.bounds.bottom, self.elev.bounds.top]

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
        elev_informations = {
            "Driver": self.elev.driver,
            "Filename": self.elev.filename,
            "Grid size": self.elev.vcrs_grid,
            "Number of band": self.elev.bands,
            "Data types": self.elev.dtype,
            "Nodata Value": self.elev.nodata,
            "Pixel interpretation": self.elev.area_or_point,
            "Pixel size": self.elev.res,
            "Width": self.elev.width,
            "Height": self.elev.height,
            "Transform": self.elev.transform,
            "Bounds": self.elev.bounds,
        }
        self.dico_to_show.append(("Elevation information", elev_informations))

        # Statistics
        list_metrics = self.config["statistics"]
        if list_metrics is not None:
            stats_elev = self.elev.get_stats(list_metrics)
            self.save_stat_as_csv(stats_elev, "stats_elev")
            self.dico_to_show.append(("Global statistics", self.floats_process(stats_elev)))
            stats_elev_mask = self.elev.get_stats(list_metrics, inlier_mask=self.inlier_mask)
            if self.inlier_mask is not None:
                self.save_stat_as_csv(stats_elev_mask, "stats_elev_mask")
                self.dico_to_show.append(("Mask statistics", self.floats_process(stats_elev_mask)))
            logging.info(f"Computing metrics on reference elevation: {list_metrics}")

        # Terrain attributes
        if self.list_attributes is not None:
            self.generate_terrain_attributes_png()
            if self.level > 1:
                self.generate_terrain_attributes_tiff()
        else:
            logging.info("Computing terrain attributes: None")

        self.create_html()

        # Remove empty folder
        for folder in self.outputs_folder.rglob("*"):
            if folder.is_dir():
                try:
                    folder.rmdir()
                except OSError:
                    pass

    def create_html(self) -> None:
        """
        Create HTML page from png files and table
        :return: None
        """

        html = "<html>\n<head><meta charset='UTF-8'><title>Topographic summary results</title></head>\n<body>\n"

        html += "<h2>Elevation Model</h2>\n"
        html += "<img src='plots/elev_map.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        if self.inlier_mask is not None:
            html += "<h2>Masked elevation Model</h2>\n"
            html += "<img src='plots/masked_elev_map.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        for title, dictionary in self.dico_to_show:
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

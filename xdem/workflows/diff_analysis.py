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
DiffAnalysis class from workflow
"""
import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from geoutils.raster import RasterType
from numpy import floating

import xdem
from xdem.workflows.schemas import COMPARE_SCHEMA
from xdem.workflows.workflows import Workflows


class DiffAnalysis(Workflows):
    """
    DiffAnalysis class from workflow
    """

    def __init__(self, config_dem: str | Dict[str, Any]) -> None:
        """
        Initialize the DiffAnalysis class
        :param config_dem: Path to a user configuration file
        """

        self.schema = COMPARE_SCHEMA

        super().__init__(config_dem)

        self.to_be_aligned_elev, tba_mask = self.generate_dem(self.config["inputs"]["to_be_aligned_elev"])
        self.reference_elev, ref_mask = self.generate_dem(self.config["inputs"]["reference_elev"])
        if self.reference_elev is None:
            self.reference_elev = self._get_reference_elevation()
            ref_mask = None
        self.generate_graph(self.reference_elev, "reference_elev_map")
        self.generate_graph(self.to_be_aligned_elev, "to_be_aligned_elev_map")

        self.inlier_mask = None
        if ref_mask is not None and tba_mask is not None:
            self.inlier_mask = tba_mask
        else:
            self.inlier_mask = ref_mask or tba_mask

    def _get_reference_elevation(self) -> float:
        """
        Get reference elevation
        """

        raise NotImplementedError("For now it doesn't working, please add a reference DEM")

    def _compute_coregistration(self) -> RasterType:
        """
        Wrapper for coregistration
        """
        if not self.config["coregistration"]["process"]:
            aligned_elev = self.to_be_aligned_elev
            logging.info("Coregistration not executed")
        else:
            coreg_extra = {}

            # Coregister
            from_str_to_fun = {
                "NuthKaab": lambda: xdem.coreg.NuthKaab(**coreg_extra),
                "DhMinimize": lambda: xdem.coreg.DhMinimize(**coreg_extra),
                "VerticalShift": lambda: xdem.coreg.VerticalShift(**coreg_extra),
                "DirectionalBias": lambda: xdem.coreg.DirectionalBias(**coreg_extra),
                "TerrainBias": lambda: xdem.coreg.TerrainBias(**coreg_extra),
            }

            coreg_steps = ["step_one", "step_two", "step_three"]
            coreg_functions = []

            for step in coreg_steps:
                config_coreg = self.config["coregistration"].get(step)
                if config_coreg:
                    method_name = config_coreg.get("method")
                    coreg_extra = config_coreg.get("extra_information", {})
                    coreg_fun = from_str_to_fun[method_name]()
                    coreg_functions.append(coreg_fun)

            my_coreg = sum(coreg_functions[1:], coreg_functions[0]) if len(coreg_functions) > 1 else coreg_functions[0]

            # Coregister
            aligned_elev = self.to_be_aligned_elev.coregister_3d(self.reference_elev, my_coreg, self.inlier_mask)
            aligned_elev.save(self.outputs_folder / "raster" / "aligned_elev.tif")

            self.dico_to_show.append(("Coregistration user configuration", self.config["coregistration"]))

            for idx, step in enumerate(coreg_steps):
                config_coreg = self.config["coregistration"].get(step)
                if config_coreg:
                    method_name = config_coreg.get("method")
                    self.dico_to_show.append(
                        (f"{method_name} inputs", self.floats_process(coreg_functions[idx].meta["inputs"]))
                    )
                    self.dico_to_show.append(
                        (f"{method_name} outputs", self.floats_process(coreg_functions[idx].meta["outputs"]))
                    )

        return aligned_elev

    def _compute_reproj(self, test_dem: str) -> None:
        """
        Compute reprojection
        :param test_dem: str value for testing the target dem
        """
        # Reproject data

        sampling = self.config["coregistration"]["sampling_source"]
        if sampling == test_dem:
            return  # No reprojection needed

        logging.info(f"Computing reprojection on {test_dem}")

        if sampling == "reference_elev":
            src, target = self.to_be_aligned_elev, self.reference_elev
            name = "to_be_aligned_elev"
            reprojected = src.reproject(target, silent=True)
            self.to_be_aligned_elev = reprojected
        elif sampling == "to_be_aligned_elev":
            src, target = self.reference_elev, self.to_be_aligned_elev
            name = "reference_elevation"
            reprojected = src.reproject(target, silent=True)
            self.reference_elev = reprojected

        if self.level > 1:
            output_path = self.outputs_folder / "raster" / f"{name}_reprojected.tif"
            reprojected.save(output_path)

    def _process_diff(self, diff: RasterType, title: str, filename: str, vmin: float, vmax: float) -> None:
        """
        Generate plot from an altitude differences DEM and save it
        :param diff: Altitude differences as DEM object
        :param title: Plot title
        :param filename: Name of output file
        :param vmin: Minimum value for colorbar
        :param vmax: Maximum value for colorbar
        """
        # diff.plot(title=title, vmin=vmin, vmax=vmax, cmap='RdBu')
        self.generate_graph(diff, filename, vmin=vmin, vmax=vmax, cmap="RdBu")

    def _get_stats(self, dem: RasterType) -> floating[Any] | dict[str, floating[Any]]:
        """
        Return a list of computed statistics chose by user or the default one.
        :param dem: DEM to process
        """
        # Compute user statistics
        list_to_compute = self.config["statistics"]
        logging.info(f"Computed statistics: {list_to_compute}")
        list_stat = dem.get_stats(list_to_compute)
        return list_stat

    def _compute_histogram(self) -> None:
        """
        Compute altitudes difference histogram.
        """
        logging.info("Compute histogram")
        plt.figure(figsize=(7, 8))
        bins = np.linspace(self.stats_before["min"], self.stats_before["max"], 300)
        plt.hist(self.diff_before.data.flatten(), bins=bins, color="g", alpha=0.6, label="Before_coregistration")
        plt.hist(self.diff_after.data.flatten().data, bins=bins, color="b", alpha=0.6, label="After_coregistration")
        plt.text(
            -50,
            50000,
            f'Before\nmed: {self.stats_before["median"]:.2f}\nnmad: {self.stats_before["nmad"]:.2f}',
            color="green",
            fontsize=12,
        )
        plt.text(
            30,
            50000,
            f'After\nmed: {self.stats_after["median"]:.2f}\nnmad: {self.stats_after["nmad"]:.2f}',
            color="blue",
            fontsize=12,
        )
        plt.title("Differences histograms", fontsize=14)
        plt.xlabel("Elev. Diff. (m)")
        plt.ylabel("Count (px)")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(self.outputs_folder / "png" / "elev_diff_histo.png")
        plt.close()

    def run(self) -> None:
        """
        Run function for the coregistration workflow
        :return: None
        """

        # Reprojection step
        self._compute_reproj("reference_elev")
        self._compute_reproj("to_be_aligned_elev")

        # Coregistration step
        aligned_elev = self._compute_coregistration()

        # Altitude differences
        for label, dem in [("before", self.to_be_aligned_elev), ("after", aligned_elev.reproject(self.reference_elev))]:
            diff = dem - self.reference_elev
            stats = diff.get_stats(["min", "max", "nmad", "median"])
            if label == "before":
                self.diff_before, self.stats_before = diff, stats
            else:
                self.diff_after, self.stats_after = diff, stats
            vmin, vmax = self.stats_before["min"], self.stats_before["max"]
            self.generate_graph(diff, f"diff_elev_{label}_coreg", vmin=vmin, vmax=vmax, cmap="RdBu")
            # self._process_diff(diff, label, f"diff_elev_{label}_coreg", vmin, vmax)

        # Statistics
        stat_items = [
            (self.reference_elev, "reference_elev", "Statistics on reference elevation", 2),
            (self.to_be_aligned_elev, "to_be_aligned_elev", "Statistics on to be aligned elevation", 2),
            (self.diff_before, "diff_elev_before_coreg", "Statistics on alti diff before coregistration", 2),
            (self.diff_after, "diff_elev_after_coreg", "Statistics on alti diff after coregistration", 2),
            (aligned_elev, "aligned_elev", "Statistics aligned DEM", 1),
        ]

        for data, fname, title, level in stat_items:
            stats = self._get_stats(data)
            if level <= self.level:
                self.save_stat_as_csv(stats, fname)
            self.dico_to_show.append((title, self.floats_process(stats)))

        if self.level > 1:
            self.diff_before.save(self.outputs_folder / "raster" / "diff_elev_before_coreg.tif")
            self.diff_after.save(self.outputs_folder / "raster" / "diff_elev_after_coreg.tif")

        # Compute altitude differences value histogram
        self._compute_histogram()

        self.create_html(self.dico_to_show)

    def create_html(self, list_dict: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Create HTML page from png files and table
        :param list_dict: list containing tuples of title and various dictionaries
        :return: None
        """
        html = "<html>\n<head><meta charset='UTF-8'><title>Qualify DEM results</title></head>\n<body>\n"

        html += "<h2>Digital Elevation Model</h2>\n"
        html += "<div style='display: flex; gap: 10px;'>\n"
        html += (
            "  <img src='png/reference_elev_map.png' alt='Image PNG' "
            "style='max-width: 100%; height: auto; width: 40%;'>\n"
        )
        html += (
            "  <img src='png/to_be_aligned_elev_map.png' alt='Image PNG' style='max-width: "
            "100%; height: auto; width: 40%;'>\n"
        )
        html += "</div>\n"

        for title, dictionary in list_dict:  # type: ignore
            html += "<div style='clear: both; margin-bottom: 30px;'>\n"
            html += f"<h2>{title}</h2>\n"
            html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            html += "<tr><th>Information</th><th>Value</th></tr>\n"
            for key, value in dictionary.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            html += "</table>\n"
            html += "</div>\n"

        html += "<h2>Altitude differences</h2>\n"
        html += "<div style='display: flex; gap: 10px;'>\n"
        html += (
            "  <img src='png/diff_elev_before_coreg.png' alt='Image PNG' style='max-width: "
            "40%; height: auto; width: 50%;'>\n"
        )
        html += (
            "  <img src='png/diff_elev_after_coreg.png' alt='Image PNG' style='max-width: "
            "40%; height: auto; width: 50%;'>\n"
        )
        html += "</div>\n"

        html += "<h2>Differences histogram</h2>\n"
        html += "<img src='png/elev_diff_histo.png' alt='Image PNG' style='max-width: 40%; height: auto;'>\n"

        html += """
             </div>
         </body>
         </html>
         """

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)

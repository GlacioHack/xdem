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
DemComparison class from workflow
"""
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from geoutils.raster import RasterType
from numpy import floating

import xdem
from xdem.workflows.workflows import Workflows


class DemComparison(Workflows):
    """
    DemComparison class from workflow
    """

    def __init__(self, config_dem: str) -> None:
        """
        Initialize the DemComparison class
        :param config_dem: Path to a user configuration file
        """

        self.schema = {
            "inputs": {
                "type": "dict",
                "schema": {
                    "reference_elev": {
                        "type": "dict",
                        "schema": {
                            "dem": {"type": "string", "required": True},
                            "nodata": {"type": ["integer", "float"], "required": False},
                            "mask": {"type": "string", "required": False},
                        },
                    },
                    "to_be_aligned_elev": {
                        "type": "dict",
                        "schema": {
                            "dem": {"type": "string", "required": True},
                            "nodata": {"type": ["integer", "float"], "required": False},
                            "mask": {"type": "string", "required": False},
                        },
                    },
                },
            },
            "outputs": {
                "type": "dict",
                "schema": {
                    "path": {"type": "string", "required": True},
                    "dem": {"type": "boolean", "required": True},
                },
            },
            "coregistration": {
                "type": "dict",
                "schema": {
                    "step_one": {
                        "type": "dict",
                        "schema": {
                            "method": {
                                "type": "string",
                                "allowed": [
                                    "nuth_and_kaab",
                                    "dh_minimize",
                                    "vertical_shift",
                                    "directional_bias",
                                    "terrain_bias",
                                ],
                                "required": True,
                            },
                            "extra_information": {"type": "dict", "required": False},
                        },
                    },
                    "step_two": {
                        "type": "dict",
                        "required": False,
                        "schema": {
                            "method": {
                                "type": "string",
                                "allowed": [
                                    "nuth_and_kaab",
                                    "dh_minimize",
                                    "vertical_shift",
                                    "directional_bias",
                                    "terrain_bias",
                                ],
                                "required": True,
                            },
                            "extra_information": {"type": "dict", "required": False},
                        },
                    },
                    "step_three": {
                        "type": "dict",
                        "required": False,
                        "schema": {
                            "method": {
                                "type": "string",
                                "allowed": [
                                    "nuth_and_kaab",
                                    "dh_minimize",
                                    "vertical_shift",
                                    "directional_bias",
                                    "terrain_bias",
                                ],
                                "required": True,
                            },
                            "extra_information": {"type": "dict", "required": False},
                        },
                    },
                    "sampling_source": {
                        "type": "string",
                        "allowed": ["reference_elev", "to_be_aligned_elev"],
                        "required": True,
                    },
                },
            },
            "statistics": {
                "type": "dict",
                "required": False,
                "schema": {
                    "list_to_compute": {"type": "list", "required": False},
                    "reference_elev": {"type": "boolean", "required": False},
                    "to_be_aligned_elev": {"type": "boolean", "required": False},
                    "alti_diff": {"type": "boolean", "required": False},
                },
            },
        }

        super().__init__(config_dem)

        self.validate_yaml()
        # Verify entry
        assert os.path.isfile(
            self.config["inputs"]["reference_elev"]["dem"]
        ), f"{self.config['inputs']['reference_elev']['dem']} does not exist"
        assert os.path.isfile(
            self.config["inputs"]["to_be_aligned_elev"]["dem"]
        ), f"{self.config['inputs']['to_be_aligned_elev']['dem']} does not exist"

        self.reference_elev, ref_mask = self.generate_dem(self.config["inputs"]["reference_elev"])
        self.to_be_aligned_dem, tba_mask = self.generate_dem(self.config["inputs"]["to_be_aligned_elev"])
        self.generate_graph(self.reference_elev, "Reference_elevation")
        self.generate_graph(self.to_be_aligned_dem, "To_be_aligned_elevation")

        self.inlier_mask = ref_mask or tba_mask

    def _compute_coregistration(self) -> RasterType:
        """
        Wrapper for coregistration
        """
        self.coreg_extra = {}
        # Coregister
        from_str_to_fun = {
            "nuth_and_kaab": lambda: xdem.coreg.NuthKaab(**self.coreg_extra),
            "dh_minimize": lambda: xdem.coreg.DhMinimize(**self.coreg_extra),
            "vertical_shift": lambda: xdem.coreg.VerticalShift(**self.coreg_extra),
            "directional_bias": lambda: xdem.coreg.DirectionalBias(**self.coreg_extra),
            "terrain_bias": lambda: xdem.coreg.TerrainBias(**self.coreg_extra),
        }

        coreg_steps = ["step_one", "step_two", "step_three"]
        coreg_functions = []

        for step in coreg_steps:
            config_coreg = self.config["coregistration"].get(step)
            if config_coreg:
                method_name = config_coreg.get("method")
                self.coreg_extra = config_coreg.get("extra_information", {})
                coreg_fun = from_str_to_fun[method_name]()
                coreg_functions.append(coreg_fun)

        my_coreg = sum(coreg_functions[1:], coreg_functions[0]) if len(coreg_functions) > 1 else coreg_functions[0]

        # Coregister
        aligned_dem = self.to_be_aligned_dem.coregister_3d(self.reference_elev, my_coreg, self.inlier_mask)
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

        return aligned_dem

    def _compute_reproj(self, test_dem: str) -> None:
        """
        Compute reprojection
        """
        # Reproject data
        src = self.config["coregistration"]["sampling_source"]

        if src == test_dem:
            logging.info(f"Computing reprojection on {test_dem}")
            src_dem = getattr(self, src)
            target_dem = getattr(self, "to_be_aligned_dem" if src == "reference_elev" else "reference_elev")

            reprojected = src_dem.reproject(target_dem, silent=True)
            setattr(self, src, reprojected)

            if self.save_dem:
                filename = f"{src}_reprojected.tif"
                reprojected.save(self.path_tiff / filename)

    def _process_diff(self, diff: RasterType, title: str, filename: str, vmin: float, vmax: float) -> None:
        """
        Generate plot from an altitude differences DEM and save it
        :param diff: Altitude differences as DEM object
        :param title: Plot title
        :param filename: Name of output file
        :param vmin: Minimum value for colorbar
        :param vmax: Maximum value for colorbar
        """
        diff.plot(title=title, vmin=vmin, vmax=vmax)
        self.generate_graph(diff, filename, vmin=vmin, vmax=vmax)

    def _get_stats_with_optional_list(self, dem: RasterType) -> floating[Any] | dict[str, floating[Any]]:
        """
        Return a list of computed statistics chose by user or the default one.
        :param dem: DEM to process
        """
        # Compute user statistics
        list_to_compute = self.config["statistics"].get("list_to_compute", [])
        logging.info(
            f"Computed statistics: {list_to_compute}"
            if list_to_compute
            else "All the metrics are computed for the statistics"
        )
        list_stat = dem.get_stats(list_to_compute) if list_to_compute else dem.get_stats()
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
        plt.savefig(self.path_png / "histo_diff.png")
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
        aligned_dem = self._compute_coregistration()

        # Altitude differences
        # Before
        self.diff_before = self.to_be_aligned_dem - self.reference_elev
        self.stats_before = self.diff_before.get_stats(["min", "max", "nmad", "median"])
        vmin, vmax = self.stats_before["min"], self.stats_before["max"]
        self.diff_before.plot(title="before", vmin=vmin, vmax=vmax)
        self._process_diff(self.diff_before, "before", "Altitude_difference_before_coregistration", vmin, vmax)
        # After
        self.diff_after = aligned_dem.reproject(self.reference_elev) - self.reference_elev
        self.stats_after = self.diff_after.get_stats(["min", "max", "nmad", "median"])
        self.diff_after.plot(title="after", vmin=vmin, vmax=vmax)
        self._process_diff(self.diff_after, "after", "Altitude_difference_after_coregistration", vmin, vmax)

        if self.save_dem:
            self.diff_before.save(self.path_tiff / "diff_before.tif")
            self.diff_after.save(self.path_tiff / "diff_after.tif")

        # Compute altitude differences value histogram
        self._compute_histogram()

        # Statistics
        # Stats on ref
        if self.config["statistics"]["reference_elev"]:
            ref_stats = self._get_stats_with_optional_list(self.reference_elev)
            self.dico_to_show.append(("Statistics on reference elevation", self.floats_process(ref_stats)))
        # Stats on tba
        if self.config["statistics"]["to_be_aligned_elev"]:
            tba_stats = self._get_stats_with_optional_list(self.to_be_aligned_dem)
            self.dico_to_show.append(("Statistics on to be aligned elevation", self.floats_process(tba_stats)))
        # Stats on alti_diff
        if self.config["statistics"]["alti_diff"]:
            alti_diff_before_stats = self._get_stats_with_optional_list(self.diff_before)
            self.dico_to_show.append(
                ("Statistics on alti diff before coregistration", self.floats_process(alti_diff_before_stats))
            )

            alti_diff_after_stats = self._get_stats_with_optional_list(self.diff_after)
            self.dico_to_show.append(
                ("Statistics on alti diff after coregistration", self.floats_process(alti_diff_after_stats))
            )

        self.create_html(self.dico_to_show)

    def create_html(self, list_dict: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Create html page from png files and table
        :param list_dict: list containing tuples of title and various dictionaries
        :return: None
        """
        html = "<html>\n<head><meta charset='UTF-8'><title>Qualify DEM results</title></head>\n<body>\n"

        html += "<h2>Digital Elevation Model</h2>\n"
        html += "<div style='display: flex; gap: 10px;'>\n"
        html += (
            "  <img src='png/Reference_elevation.png' alt='Image PNG' "
            "style='max-width: 100%; height: auto; width: 40%;'>\n"
        )
        html += (
            "  <img src='png/To_be_aligned_elevation.png' alt='Image PNG' style='max-width: "
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
            "  <img src='png/Altitude_difference_before_coregistration.png' alt='Image PNG' style='max-width: "
            "40%; height: auto; width: 50%;'>\n"
        )
        html += (
            "  <img src='png/Altitude_difference_after_coregistration.png' alt='Image PNG' style='max-width: "
            "40%; height: auto; width: 50%;'>\n"
        )
        html += "</div>\n"

        html += "<h2>Differences histogram</h2>\n"
        html += "<img src='png/histo_diff.png' alt='Image PNG' style='max-width: 40%; height: auto;'>\n"

        html += """
             </div>
         </body>
         </html>
         """

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)

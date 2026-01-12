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
Accuracy class from workflow
"""

import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from geoutils.raster import RasterType
from numpy import floating

import xdem
from xdem._misc import import_optional
from xdem.workflows.schemas import ACCURACY_SCHEMA
from xdem.workflows.workflows import Workflows


class Accuracy(Workflows):
    """
    Accuracy class, inherits from the Workflow class to apply a workflow.
    """

    def __init__(self, config_dem: str | Dict[str, Any], output: str | None = None) -> None:
        """
        Initialization with configuration file.

        :param config_dem: Path to user configuration file.
        """

        yaml = import_optional("yaml", package_name="pyyaml")

        self.schema = ACCURACY_SCHEMA
        self.elapsed: float | None = None
        self.df_stats: pd.DataFrame | None = None

        super().__init__(config_dem, output)

        self.compute_coreg = self.config["coregistration"]["process"]

        if not self.compute_coreg:
            del self.config["coregistration"]["step_one"]

        yaml_str = yaml.dump(self.config, allow_unicode=True, Dumper=self.NoAliasDumper)
        Path(self.outputs_folder / "used_config.yaml").write_text(yaml_str, encoding="utf-8")

        self.config = self.remove_none(self.config)  # type: ignore

    def _load_data(self) -> None:
        """Load data."""

        self.to_be_aligned_elev, tba_mask, tba_path_mask = self.load_dem(self.config["inputs"]["to_be_aligned_elev"])
        self.reference_elev, ref_mask, ref_mask_path = self.load_dem(self.config["inputs"].get("reference_elev", None))
        if self.reference_elev is None:
            self.reference_elev = self._get_reference_elevation()
        self.generate_plot(
            self.reference_elev,
            title="Reference DEM",
            filename="reference_elev_map",
            cmap="terrain",
            cbar_title="Elevation (m)",
        )
        self.generate_plot(
            self.to_be_aligned_elev,
            title="To-be-aligned DEM",
            filename="to_be_aligned_elev_map",
            cmap="terrain",
            cbar_title="Elevation (m)",
        )

        self.inlier_mask = None
        if ref_mask is not None and tba_mask is not None:
            self.inlier_mask = tba_mask
            path_mask = tba_path_mask
        else:
            self.inlier_mask = ref_mask or tba_mask
            path_mask = ref_mask_path or tba_path_mask

        if self.inlier_mask is not None:
            self.generate_plot(
                self.to_be_aligned_elev,
                title="Masked (inlier) terrain",
                filename="masked_elev_map",
                mask_path=path_mask,
                cmap="terrain",
                cbar_title="Elevation (m)",
            )

    def _get_reference_elevation(self) -> float:
        """
        Get reference elevation.
        """

        raise NotImplementedError("This is not implemented, add a reference DEM")

    def _compute_coregistration(self) -> RasterType:
        """
        Wrapper for coregistration.
        """

        coreg_steps = ["step_one", "step_two", "step_three"]
        coreg_functions = []

        method_map = {
            "NuthKaab": xdem.coreg.NuthKaab,
            "DhMinimize": xdem.coreg.DhMinimize,
            "VerticalShift": xdem.coreg.VerticalShift,
            "DirectionalBias": xdem.coreg.DirectionalBias,
            "TerrainBias": xdem.coreg.TerrainBias,
            "LZD": xdem.coreg.LZD,
        }

        for step in coreg_steps:
            config_coreg = self.config["coregistration"].get(step)
            if config_coreg:
                method_name = config_coreg.get("method")
                coreg_extra = config_coreg.get("extra_information", {})
                coreg_fun = partial(method_map[method_name], **coreg_extra)
                coreg_functions.append(coreg_fun())

        my_coreg = sum(coreg_functions[1:], coreg_functions[0]) if len(coreg_functions) > 1 else coreg_functions[0]

        # Coregister
        aligned_elev = self.to_be_aligned_elev.coregister_3d(
            self.reference_elev, my_coreg, self.inlier_mask, random_state=42
        )
        aligned_elev.to_file(self.outputs_folder / "rasters" / "aligned_elev.tif")

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

    def _prepare_datas_for_coreg(self) -> None:
        """
        Compute reprojection.
        """
        sampling_source = self.config["inputs"]["sampling_grid"]

        # Reprojection
        if sampling_source == "reference_elev":
            crs_utm = self.reference_elev.get_metric_crs()
        else:
            crs_utm = self.to_be_aligned_elev.get_metric_crs()

        logging.info("Computing reprojection")
        if not crs_utm.is_geographic:
            logging.info(f"CRS not geographic: data reprojection with {crs_utm}")
            self.to_be_aligned_elev = self.to_be_aligned_elev.reproject(crs=crs_utm)
            self.reference_elev = self.reference_elev.reproject(crs=crs_utm)

        if sampling_source == "reference_elev":
            self.to_be_aligned_elev = self.to_be_aligned_elev.reproject(self.reference_elev, silent=True)
        elif sampling_source == "to_be_aligned_elev":
            self.reference_elev = self.reference_elev.reproject(self.to_be_aligned_elev, silent=True)

        # Intersection
        logging.info("Computing intersection")
        coord_intersection = self.reference_elev.intersection(self.to_be_aligned_elev)
        if sampling_source == "reference_elev":
            self.reference_elev = self.reference_elev.crop(coord_intersection)
            self.generate_plot(
                self.to_be_aligned_elev,
                title="Cropped reference DEM",
                filename="cropped_reference_elev_map",
                cmap="terrain",
                cbar_title="Elevation (m)",
            )
        else:
            self.to_be_aligned_elev = self.to_be_aligned_elev.crop(coord_intersection)
            self.generate_plot(
                self.to_be_aligned_elev,
                title="Cropped to-be-aligned DEM",
                filename="cropped_to_be_aligned_elev_map",
                cmap="terrain",
                cbar_title="Elevation (m)",
            )

        if self.level > 1:
            self.reference_elev.to_file(self.outputs_folder / "rasters" / "reference_elev_reprojected.tif")
            self.to_be_aligned_elev.to_file(self.outputs_folder / "rasters" / "to_be_aligned_elev_reprojected.tif")

    def _get_stats(self, dem: RasterType, name_of_data: str = "") -> floating[Any] | dict[str, floating[Any]]:
        """
        Return a list of computed statistics chose by user or the default one.

        :param dem: Input DEM.
        :param name_of_data: Logging string.
        """
        # Compute user statistics
        dict_stats_aliased = {}
        list_to_compute = self.config["statistics"]
        if list_to_compute is not None:
            logging.info(f"Computing statistics on {name_of_data}: {list_to_compute}")
            dict_stats = dem.get_stats(list_to_compute)

            # Aliases for nicer CSV headers
            aliases = {
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
                "std": "STD",
                "standarddeviation": "Standard deviation",
                "validcount": "Valid count",
                "totalcount": "Total count",
                "percentagevalidpoints": "Percentage valid points",
            }

            dict_stats_aliased = {aliases.get(k, k): v for k, v in dict_stats.items()}

        return dict_stats_aliased

    def _compute_histogram(self) -> None:
        """
        Compute altitudes difference histogram.
        """

        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        logging.info("Computing histogram on altitude difference")
        plt.figure(figsize=(12, 6))
        bins = np.linspace(self.stats_before["min"], self.stats_before["max"], 300)
        plt.xlim((-4 * np.std(self.diff_before), 4 * np.std(self.diff_before)))
        plt.hist(self.diff_before.data.flatten(), bins=bins, color="g", alpha=0.5, label="Before coregistration")
        plt.hist(self.diff_after.data.flatten().data, bins=bins, color="b", alpha=0.5, label="After coregistration")
        ax = plt.gca()
        plt.text(
            0.2,
            0.5,
            f'Before:\nmedian = {self.stats_before["median"]:.2f}\nnmad = {self.stats_before["nmad"]:.2f}',
            color="g",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        plt.text(
            0.8,
            0.5,
            f'After:\nmedian = {self.stats_after["median"]:.2f}\nnmad = {self.stats_after["nmad"]:.2f}',
            color="b",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        plt.title("Histogram of elevation differences\nbefore and after coregistration")
        plt.xlabel("Elevation differences (m)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(False)
        plt.savefig(self.outputs_folder / "plots" / "elev_diff_histo.png")
        plt.close()

    def run(self) -> None:
        """
        Run function for the coregistration workflow.

        :return: None
        """

        t0 = time.time()

        self._load_data()

        # Reprojection step
        if "sampling_grid" in self.config["inputs"]:
            print(self.config["inputs"]["sampling_grid"])
            self._prepare_datas_for_coreg()

        if self.compute_coreg:
            # Coregistration step
            aligned_elev = self._compute_coregistration()
        else:
            logging.info("Coregistration not executed, returned to_be_aligned_elev")
            aligned_elev = self.to_be_aligned_elev

        output_grid = self.config["outputs"]["output_grid"]
        ref_elev = self.reference_elev if output_grid == "reference_elev" else self.to_be_aligned_elev

        vmin = vmax = None

        if self.compute_coreg:
            diff_pairs = [("before", self.to_be_aligned_elev), ("after", aligned_elev.reproject(ref_elev))]
        else:
            diff_pairs = [("", self.to_be_aligned_elev)]

        for label, dem in diff_pairs:
            diff = dem - ref_elev
            stats_keys = ["min", "max", "nmad", "median"]
            stats = diff.get_stats(stats_keys)

            if label == "before":
                self.diff_before, self.stats_before = diff, stats
                vmin, vmax = -(stats["median"] + 3 * stats["nmad"]), stats["median"] + 3 * stats["nmad"]
            elif label == "after":
                self.diff_after, self.stats_after = diff, stats
            else:
                self.diff = diff
                vmin, vmax = -(stats["median"] + 3 * stats["nmad"]), (stats["median"] + 3 * stats["nmad"])

            suffix = f"_elev_{label}_coreg_map" if label else "_elev"
            self.generate_plot(
                diff,
                title=f"Difference\n{label} coregistration",
                filename=f"diff{suffix}",
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu",
                cbar_title="Elevation differences (m)",
            )

        if self.compute_coreg:
            stat_items = [
                (self.reference_elev, "reference_elev", "Reference elevation", 2),
                (self.to_be_aligned_elev, "to_be_aligned_elev", "To-be-aligned elevation", 2),
                (aligned_elev, "aligned_elev", "Aligned elevation", 1),
                (
                    self.diff_before,
                    "diff_elev_before_coreg",
                    "Difference before coreg",
                    1,
                ),
                (
                    self.diff_after,
                    "diff_elev_after_coreg",
                    "Difference after coreg",
                    1,
                ),
            ]
        else:
            stat_items = [
                (self.reference_elev, "reference_elev", "Reference elevation", 2),
                (self.to_be_aligned_elev, "to_be_aligned_elev", "To-be-aligned elevation", 2),
                (self.diff, "diff_elev", "Elevation difference", 2),
            ]

        list_df_var = []
        for i, (data, fname, title, level) in enumerate(stat_items):
            if (level > self.level and self.compute_coreg) or self.config["statistics"] is None:
                continue
            stats = self._get_stats(data, fname)
            self.save_stat_as_csv(stats, fname)  # type: ignore

            df = pd.DataFrame(data=stats, index=[i])
            df.insert(loc=0, column="Data", value=[title])
            list_df_var.append(df)

        if len(list_df_var) > 0:
            df_stats = pd.concat(list_df_var)
        else:
            df_stats = None
        self.df_stats = df_stats

        if self.compute_coreg:
            self._compute_histogram()
            if self.level > 1:
                self.diff_before.to_file(self.outputs_folder / "rasters" / "diff_elev_before_coreg_map.tif")
                self.diff_after.to_file(self.outputs_folder / "rasters" / "diff_elev_after_coreg_map.tif")

        t1 = time.time()
        self.elapsed = t1 - t0

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
        Create HTML page from png files and table.

        :param list_dict: List containing tuples of title and various dictionaries.

        :return: None
        """
        html = "<html>\n<head><meta charset='UTF-8'><title>Qualify elevation results</title></head>\n<body>\n"

        # Title and version/date/time summary
        html += "<h1>Accuracy assessment report — xDEM</h1>\n"

        html += f"<p>xDEM version: {xdem.__version__}</p>"
        html += f"<p>Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>"
        html += f"<p>Computing time: {self.elapsed:.2f} seconds</p>"

        # Plot input elevation data
        html += "<h2>Elevation datasets</h2>\n"
        html += "<div style='display: flex; gap: 10px;'>\n"
        html += (
            "  <img src='plots/reference_elev_map.png' alt='Image PNG' "
            "style='max-width: 50%; height: auto; width: 50%;'>\n"
        )
        html += (
            "  <img src='plots/to_be_aligned_elev_map.png' alt='Image PNG' style='max-width: "
            "50%; height: auto; width: 50%;'>\n"
        )
        html += "</div>\n"

        def format_values(val: Any) -> Any:
            """Format values for the dictionary."""
            if isinstance(val, float):
                return np.format_float_positional(val)
            elif callable(val):
                return val.__name__
            else:
                return str(val)

        # Metadata: Inputs, coregistration
        for title, dictionary in list_dict:  # type: ignore
            html += "<div style='clear: both; margin-bottom: 30px;'>\n"
            html += f"<h2>{title}</h2>\n"
            html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            html += "<tr><th>Information</th><th>Value</th></tr>\n"
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = {k: format_values(v) for k, v in value.items()}
                html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            html += "</table>\n"
            html += "</div>\n"

        # Statistics table:
        if self.df_stats is not None:
            html += "<h2>Statistics</h2>\n"
            html += self.df_stats.to_html(index=False)

        # Coregistration: Add elevation difference plot and histograms before/after
        if self.compute_coreg:
            html += "<h2>Elevation differences</h2>\n"
            html += "<div style='display: flex; gap: 10px;'>\n"
            html += (
                "  <img src='plots/diff_elev_before_coreg_map.png' alt='Image PNG' style='max-width: "
                "50%; height: auto; width: 50%;'>\n"
            )
            html += (
                "  <img src='plots/diff_elev_after_coreg_map.png' alt='Image PNG' style='max-width: "
                "50%; height: auto; width: 50%;'>\n"
            )
            html += "</div>\n"

            html += "<h2>Differences histogram</h2>\n"
            html += "<img src='plots/elev_diff_histo.png' alt='Image PNG' style='max-width: 100%; height: auto;'>\n"

        else:
            html += "<h2>Elevation differences</h2>\n"
            html += "<div style='display: flex; gap: 10px;'>\n"
            html += (
                "  <img src='plots/diff_elev.png' alt='Image PNG' style='max-width: "
                "40%; height: auto; width: 50%;'>\n"
            )
            html += "</div>\n"

        html += """
             </div>
         </body>
         </html>
         """

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)

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
from xdem.workflows.workflows import _ALIAS, Workflows


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
        else:
            if self.config["inputs"]["sampling_grid"] is None:
                raise ValueError(
                    'In case of a coregistration process, "sampling grid" must be set to '
                    '"reference_elev" or "to_be_aligned_elev"'
                )

        yaml_str = yaml.dump(self.config, allow_unicode=True, Dumper=self.NoAliasDumper)
        Path(self.outputs_folder / "used_config.yaml").write_text(yaml_str, encoding="utf-8")

        self.config = self.remove_none(self.config)  # type: ignore

    def _load_data(self) -> tuple[float, float]:
        """
        Load data

        :return vmin, vmax: to plot elevation data with the same scale
        """
        self.reference_elev, ref_mask, ref_mask_path = self.load_dem(self.config["inputs"].get("reference_elev", None))
        self.to_be_aligned_elev, tba_mask, tba_path_mask = self.load_dem(self.config["inputs"]["to_be_aligned_elev"])
        if self.reference_elev is None:
            self.reference_elev = self._get_reference_elevation()

        vmin = float(min(np.nanpercentile(self.reference_elev, q=5), np.nanpercentile(self.to_be_aligned_elev, q=5)))
        vmax = float(max(np.nanpercentile(self.reference_elev, q=95), np.nanpercentile(self.to_be_aligned_elev, q=95)))

        self.generate_plot(
            dem=self.reference_elev,
            title="Reference elevation",
            filename="inputs",
            dem_right=self.to_be_aligned_elev,
            title_dem_right="To-be-aligned elevation",
            vmin=vmin,
            vmax=vmax,
            cbar_title=f"Elevation ({self.reference_elev.crs.linear_units})",
        )
        if ref_mask is not None or tba_mask is not None:
            if ref_mask is not None:
                inlier_mask_crop = ref_mask.reproject(self.reference_elev).crop(self.reference_elev)
                self.reference_elev.set_mask(~inlier_mask_crop)
            if tba_mask is not None:
                inlier_mask_crop = tba_mask.reproject(self.to_be_aligned_elev).crop(self.to_be_aligned_elev)
                self.to_be_aligned_elev.set_mask(~inlier_mask_crop)

            self.generate_plot(
                self.reference_elev,
                title="Masked terrain for reference elevation",
                filename="masked_elev_map",
                dem_right=self.to_be_aligned_elev,
                title_dem_right="Masked terrain for to-be-aligned elevation",
                vmin=vmin,
                vmax=vmax,
                cbar_title=f"Elevation ({self.reference_elev.crs.linear_units})",
            )

        return vmin, vmax

    def _get_reference_elevation(self) -> float:
        """
        Get reference elevation.
        """

        raise NotImplementedError("This is not implemented, add a reference elevation")

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
                print(method_name)
                coreg_extra = config_coreg.get("extra_information", {})
                coreg_fun = partial(method_map[method_name], **coreg_extra)
                coreg_functions.append(coreg_fun())
        my_coreg = sum(coreg_functions[1:], coreg_functions[0]) if len(coreg_functions) > 1 else coreg_functions[0]

        # Coregister
        aligned_elev = self.to_be_aligned_elev.coregister_3d(self.reference_elev, my_coreg, random_state=42)
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

    def _prepare_datas(self, vmin: float, vmax: float) -> None:
        """
        Compute reprojection.

        :param vmin: to plot elevation data with the same scale
        :param vmax: to plot elevation data with the same scale
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
            self.to_be_aligned_elev = self.to_be_aligned_elev.crop(coord_intersection)
            self.generate_plot(
                self.to_be_aligned_elev,
                title="Preprocessed to-be-aligned elevation",
                filename="preprocessed_to_be_aligned_elev_map",
                vmin=vmin,
                vmax=vmax,
                cbar_title=f"Elevation ({self.to_be_aligned_elev.crs.linear_units})",
            )
        else:
            self.reference_elev = self.reference_elev.crop(coord_intersection)
            self.generate_plot(
                self.reference_elev,
                title="Preprocessed reference elevation",
                filename="preprocessed_reference_elev_map",
                vmin=vmin,
                vmax=vmax,
                cbar_title=f"Elevation ({self.reference_elev.crs.linear_units})",
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
        print("list_to_compute", list_to_compute)

        if list_to_compute is not None:
            logging.info(f"Computing statistics on {name_of_data}: {list_to_compute}")
            dict_stats = dem.get_stats(list_to_compute)
            dict_stats_aliased = {_ALIAS.get(k, k): v for k, v in dict_stats.items()}

        return dict_stats_aliased

    def _compute_histogram(self) -> None:
        """
        Compute altitudes difference histogram.
        """

        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        logging.info("Computing histogram on altitude difference")

        # Force figsize with the same size as generate_plot function
        plt.figure(figsize=[6.4, 2.34])
        size_font = 6
        plt.rc("font", size=size_font)
        plt.rc("axes", titlesize=size_font)
        plt.rc("axes", labelsize=size_font)
        plt.rc("xtick", labelsize=size_font)
        plt.rc("ytick", labelsize=size_font)
        plt.rc("legend", fontsize=size_font)
        plt.rc("figure", titlesize=size_font)

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
        plt.xlabel(f"Elevation differences ({self.reference_elev.crs.linear_units})")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(False)
        plt.savefig(self.outputs_folder / "plots" / "elev_diff_histo.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _get_plot_differences_with_profiles(self, dem_diff: RasterType) -> None:
        """
        Show a plot of an alimetric difference and save it if
        specified into the config file

        :param dem_diff: Altimetric difference (as DEM object)
        """

        import_optional("matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.gridspec import GridSpec

        le90 = dem_diff.get_stats("LE90")
        median = dem_diff.get_stats("Median")

        # données raster (2D)
        data = dem_diff.data
        ny, nx = data.shape

        # Initial min/max for mean profiles
        profile_cols = data.mean(axis=0)
        profile_cols_stats = [profile_cols.min(), profile_cols.max()]
        profile_rows = data.mean(axis=1)
        profile_rows_stats = [profile_rows.min(), profile_rows.max()]

        # Keep profiles with at least more than 50% of valid values
        nb_valid_rows = data.count(axis=1)
        nb_valid_cols = data.count(axis=0)
        min_valid_rows = data.shape[1] / 2.0
        min_valid_cols = data.shape[0] / 2.0
        # Update profiles values according to valid values
        profile_rows = np.ma.masked_where(nb_valid_rows < min_valid_rows, data.mean(axis=1))
        profile_cols = np.ma.masked_where(nb_valid_cols < min_valid_cols, data.mean(axis=0))

        # Z-score application
        zscore_mask_cols = np.abs(profile_cols - np.mean(profile_cols)) >= (np.std(profile_cols) * 2)
        profile_cols[zscore_mask_cols] = np.nan

        zscore_mask_rows = np.abs(profile_rows - np.mean(profile_rows)) >= (np.std(profile_rows) * 2)
        profile_rows[zscore_mask_rows] = np.nan

        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = GridSpec(2, 3, width_ratios=[1.2, 4, 0.3], height_ratios=[1.2, 4])  # 1 pour colonne colorbar

        ax_top = fig.add_subplot(gs[0, 1])
        ax_left = fig.add_subplot(gs[1, 0])
        ax_map = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[1, 2])

        # alti diff initial
        cmap = LinearSegmentedColormap.from_list("blue_yellow_red", ["#2166ac", "#ffffbf", "#b2182b"])
        im = ax_map.imshow(
            data, cmap=cmap, vmin=median - le90 / 2, vmax=median + le90 / 2, interpolation="none", aspect="equal"
        )
        ax_map.set_adjustable("datalim")
        ax_map.set_xlabel("Column index")

        fig.colorbar(im, cax=cax).set_label("Δh [m]")

        ax_map.text(0.5, -0.12, "Altimetric difference [m]", transform=ax_map.transAxes, ha="center", va="top")

        # ---- Profil colonnes ----
        x = np.arange(nx)
        ax_top.plot(x, profile_cols, color="black")
        ax_top.set_xlim(ax_map.get_xlim())
        ax_top.yaxis.tick_left()
        ax_top.yaxis.set_label_position("left")
        ax_top.set_xlabel(
            f"Mean along columns [m] - "
            f"Min {np.round(profile_cols_stats[0], 2)}/Max {np.round(profile_cols_stats[1], 2)}"
        )
        ax_top.xaxis.set_label_position("top")

        # ---- Profil lignes ----
        y = np.arange(ny)
        ax_left.plot(profile_rows, y, color="black")
        ax_left.set_ylim(ax_map.get_ylim())
        ax_left.invert_xaxis()
        ax_left.yaxis.tick_left()
        ax_left.yaxis.set_label_position("left")
        ax_left.set_ylabel("Line index")
        ax_left.set_xlabel(
            f"Mean along lines [m] - Min {np.round(profile_rows_stats[0], 2)}/Max {np.round(profile_rows_stats[1], 2)}"
        )

        # plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.show()

    def run(self) -> None:
        """
        Run function for the coregistration workflow.

        :return: None
        """

        t0 = time.time()

        vmin, vmax = self._load_data()

        # Reprojection step
        if "sampling_grid" in self.config["inputs"]:
            self._prepare_datas(vmin, vmax)

        if self.compute_coreg:
            # Coregistration step
            aligned_elev = self._compute_coregistration()
        else:
            logging.info("Coregistration not executed, returned to_be_aligned_elev")
            aligned_elev = self.to_be_aligned_elev

        output_grid = self.config["outputs"]["output_grid"]
        ref_elev = self.reference_elev if output_grid == "reference_elev" else self.to_be_aligned_elev
        stats_keys = ["min", "max", "nmad", "median"]

        if self.compute_coreg:

            self.diff_before = self.to_be_aligned_elev - ref_elev
            self.stats_before = self.diff_before.get_stats(stats_keys)

            self.diff_after = aligned_elev.reproject(ref_elev) - ref_elev
            self.stats_after = self.diff_after.get_stats(stats_keys)

            self._get_plot_differences_with_profiles(self.diff_after)

            vmin_diff = min(
                -(self.stats_before["median"] + 3 * self.stats_before["nmad"]),
                -(self.stats_after["median"] + 3 * self.stats_after["nmad"]),
            )
            vmax_diff = max(
                self.stats_before["median"] + 3 * self.stats_before["nmad"],
                self.stats_after["median"] + 3 * self.stats_after["nmad"],
            )

            self.generate_plot(
                dem=self.diff_before,
                title="Elevation difference before coregistration",
                filename="diff_elev_diff_coreg_map",
                dem_right=self.diff_after,
                title_dem_right="Elevation difference after coregistration",
                vmin=vmin_diff,
                vmax=vmax_diff,
                cmap="RdBu",
                cbar_title=f"Elevation differences ({self.diff_before.crs.linear_units})",
            )

        else:
            self.diff = self.to_be_aligned_elev - ref_elev
            self.stats = self.diff.get_stats(stats_keys)
            vmin, vmax = -(self.stats["median"] + 3 * self.stats["nmad"]), self.stats["median"] + 3 * self.stats["nmad"]
            self.generate_plot(
                self.diff,
                title="Elevation difference without coregistration",
                filename="diff_elev_without_coreg_map",
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu",
                cbar_title=f"Elevation differences ({self.diff.crs.linear_units})",
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
                (self.diff, "diff_elev_without_coreg", "Elevation difference without coregistration", 2),
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
            df_stats.set_index("Data", inplace=True)
        else:
            df_stats = None
        self.df_stats = df_stats

        if self.compute_coreg:
            self._compute_histogram()
            if self.level > 1:
                self.diff_before.to_file(self.outputs_folder / "rasters" / "diff_elev_before_coreg_map.tif")
                self.diff_after.to_file(self.outputs_folder / "rasters" / "diff_elev_after_coreg_map.tif")
        else:
            if self.level > 1:
                self.diff.to_file(self.outputs_folder / "rasters" / "diff_elev_without_coreg_map.tif")

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
        html += "<h2>Elevation inputs</h2>\n"
        html += "<img src='plots/inputs.png' alt='Image PNG' style='width: 100%; height: auto;'>\n"

        if (
            "path_to_mask" in self.config["inputs"]["reference_elev"]
            or "path_to_mask" in self.config["inputs"]["to_be_aligned_elev"]
        ):
            html += "<h2>Masked elevation data</h2>\n"
            html += "<img src='plots/masked_elev_map.png' alt='Image PNG' style='width: 100%; height: auto;'>\n"

        def format_values(val: Any) -> Any:
            """Format values for the dictionary."""
            if isinstance(val, float):
                return np.format_float_positional(val)
            elif callable(val):
                return val.__name__
            else:
                return str(val)

        def print_dict(title: str, dictionary: dict[str, Any]) -> str:
            div_html = "<div style='clear: both; margin-bottom: 30px;'>\n"
            div_html += f"<h2>{title}</h2>\n"
            div_html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            div_html += "<tr><th>Information</th><th>Value</th></tr>\n"
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = {k: format_values(v) for k, v in value.items()}
                div_html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            div_html += "</table>\n"
            div_html += "</div>\n"
            return div_html

        # Metadata: Inputs
        inputs_information = list_dict[0]
        html += print_dict(inputs_information[0], inputs_information[1])

        # Plot preprocessed data if did
        if "sampling_grid" in self.config["inputs"] and self.config["inputs"]["sampling_grid"] is not None:
            if self.config["inputs"]["sampling_grid"] == "reference_elev":
                preprocessed_data = "plots/preprocessed_to_be_aligned_elev_map.png"
            else:
                preprocessed_data = "plots/preprocessed_reference_elev_map.png"

            html += "<h2>Preprocessed elevation data</h2>\n"
            html += "<img src='" + preprocessed_data + "' alt='Image PNG' style='width: 100%; height: auto;'>\n"

        # Metadata: Inputs
        for title, dictionary in list_dict[1:]:  # type: ignore
            html += print_dict(title, dictionary)

        # Statistics table:
        if self.df_stats is not None:
            html += "<h2>Statistics</h2>\n"
            html += "<table border='1' cellspacing='0' cellpadding='5'>\n"
            # Plot one stat by row
            df_cols = "".join([f'<td style="font-weight:bold">{col}</td>' for col in self.df_stats.T.columns])
            html += f'<tr><td style="font-weight:bold">Data</td>{df_cols}</tr>\n'
            for key, value in self.df_stats.T.iterrows():
                df_values = "".join([f"<td>{self.format_values_stats(key, val)}</td>" for val in value.values])
                html += f"<tr><td>{key}</td>{df_values}</tr>\n"
            html += "</table>\n"

        # Coregistration: Add elevation difference plot and histograms before/after
        if self.compute_coreg:
            html += "<h2>Elevation differences</h2>\n"
            html += "<img src='plots/diff_elev_diff_coreg_map.png' alt='Image PNG' style='width: 100%; height: auto'>\n"

            html += "<h2>Differences histogram</h2>\n"
            html += "<img src='plots/elev_diff_histo.png' alt='Image PNG' style='width: 100%; height: auto'>\n"

        else:
            html += "<h2>Elevation differences</h2>\n"
            html += (
                "<img src='plots/diff_elev_without_coreg_map.png' alt='Image PNG' style='width: 100%; height: auto'>\n"
            )
        html += """
         </body>
         </html>
         """

        with open(self.outputs_folder / "report.html", "w", encoding="utf-8") as f:
            f.write(html)

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
Test DiffAnalysis class
"""
# mypy: disable-error-code=no-untyped-def
from pathlib import Path

import geoutils as gu
import pytest

import xdem
from xdem.workflows import DiffAnalysis
from xdem.workflows.workflows import Workflows


def test_init_diff_analysis(get_compare_inputs_config, tmp_path, list_default_terrain_attributes):
    """
    Test initialization of DiffAnalysis class
    """
    user_config = get_compare_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = DiffAnalysis(user_config)

    assert isinstance(workflows, Workflows)
    assert isinstance(workflows, DiffAnalysis)
    assert Path(tmp_path / "png").joinpath("reference_elev_map.png").exists()
    assert Path(tmp_path / "png").joinpath("to_be_aligned_elev_map.png").exists()
    assert Path(tmp_path / "png").joinpath("reference_elev_map.png").exists()
    dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    mask = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
    inlier_mask = ~mask.create_mask(dem)
    assert workflows.inlier_mask == inlier_mask


@pytest.mark.skip("Not implemented")
def test__get_reference_elevation():
    """
    Test _get_reference_elevation function
    """


@pytest.mark.skip("Not implemented")
def test__compute_coregistration():
    """
    Test _compute_coregistration function
    """


@pytest.mark.skip("Not implemented")
def test__compute_reproj():
    """
    Test _compute_reproj function
    """


@pytest.mark.skip("Not implemented")
def test__process_diff():
    """
    Test _process_diff function
    """


def test__get_stats(get_compare_inputs_config, tmp_path):
    """
    Test _get_stats function
    """
    user_config = get_compare_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = DiffAnalysis(user_config)

    dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    assert workflows._get_stats(dem) == dem.get_stats(
        [
            "mean",
            "median",
            "max",
            "min",
            "sum",
            "sumofsquares",
            "90thpercentile",
            "le90",
            "nmad",
            "rmse",
            "std",
            "standarddeviation",
            "validcount",
            "totalcount",
            "percentagevalidpoints",
        ]
    )


@pytest.mark.skip("Not implemented")
def test__compute_histogram(get_compare_inputs_config, tmp_path):
    """
    Test _compute_histogram function
    """


@pytest.mark.parametrize(
    "level",
    [1, 2],
)
def test_run(get_compare_inputs_config, tmp_path, level):
    """
    Test run function
    """

    user_config = get_compare_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    workflows = DiffAnalysis(user_config)
    workflows.run()

    assert Path(tmp_path / "csv").joinpath("aligned_elev_stats.csv").exists()

    assert Path(tmp_path / "png").joinpath("diff_elev_after_coreg.png").exists()
    assert Path(tmp_path / "png").joinpath("diff_elev_before_coreg.png").exists()
    assert Path(tmp_path / "png").joinpath("elev_diff_histo.png").exists()
    assert Path(tmp_path / "png").joinpath("masked elevation.png").exists()
    assert Path(tmp_path / "png").joinpath("reference_elev_map.png").exists()
    assert Path(tmp_path / "png").joinpath("to_be_aligned_elev_map.png").exists()

    assert Path(tmp_path / "raster").joinpath("aligned_elev.tif").exists()

    assert Path(tmp_path).joinpath("report.html").exists()
    # sometimes the PDF creation fails for no reason
    # assert Path(tmp_path).joinpath("report.pdf").exists()
    assert Path(tmp_path).joinpath("used_config.yaml").exists()

    csv_files = [
        "diff_elev_after_coreg_stats.csv",
        "diff_elev_before_coreg_stats.csv",
        "reference_elev_stats.csv",
        "to_be_aligned_elev_stats.csv",
    ]

    raster_files = ["diff_elev_after_coreg.tif", "diff_elev_before_coreg.tif", "to_be_aligned_elev_reprojected.tif"]

    if level == 1:
        for file in csv_files:
            assert not (Path(tmp_path) / "csv" / file).exists()
        for file in raster_files:
            assert not (Path(tmp_path) / "raster" / file).exists()

    if level == 2:
        for file in csv_files:
            assert (Path(tmp_path) / "csv" / file).exists()
        for file in raster_files:
            assert (Path(tmp_path) / "raster" / file).exists()


@pytest.mark.skip("Not implemented")
def test_create_html():
    """
    Test create_html function
    """

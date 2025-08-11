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

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_init_diff_analysis(get_diffanalysis_object_with_run, tmp_path):
    """
    Test initialization of DiffAnalysis class
    """
    workflows = get_diffanalysis_object_with_run

    assert isinstance(workflows, Workflows)
    assert isinstance(workflows, DiffAnalysis)
    assert Path(tmp_path / "png").joinpath("reference_elev_map.png").exists()
    assert Path(tmp_path / "png").joinpath("to_be_aligned_elev_map.png").exists()
    assert Path(tmp_path / "png").joinpath("reference_elev_map.png").exists()
    dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    mask = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
    inlier_mask = ~mask.create_mask(dem)
    assert workflows.inlier_mask == inlier_mask


def test__get_reference_elevation(get_diffanalysis_inputs_config, tmp_path):
    """
    Test _get_reference_elevation function
    """

    user_config = get_diffanalysis_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = DiffAnalysis(user_config)

    with pytest.raises(NotImplementedError, match="For now it doesn't working, please add a reference DEM"):
        workflows._get_reference_elevation()

    user_config = get_diffanalysis_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["inputs"]["reference_elev"] = None

    with pytest.raises(NotImplementedError, match="For now it doesn't working, please add a reference DEM"):
        _ = DiffAnalysis(user_config)


@pytest.mark.skip("Not implemented")
def test__compute_coregistration():
    """
    Test _compute_coregistration function
    """


def test__compute_reproj(get_diffanalysis_inputs_config, tmp_path):
    """
    Test _compute_reproj function
    """
    user_config = get_diffanalysis_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["coregistration"] = {"process": False}
    user_config["coregistration"]["sampling_grid"] = "to_be_aligned_elev"
    workflows = DiffAnalysis(user_config)
    workflows.run()
    src, target = workflows.reference_elev, workflows.to_be_aligned_elev
    gt_reprojected = src.reproject(target, silent=True)

    assert workflows.reference_elev == gt_reprojected


def test__get_stats(get_diffanalysis_inputs_config, tmp_path):
    """
    Test _get_stats function
    """
    user_config = get_diffanalysis_inputs_config
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


def test__compute_histogram(get_diffanalysis_object_with_run, tmp_path):
    """
    Test _compute_histogram function
    """

    _ = get_diffanalysis_object_with_run

    assert Path(tmp_path / "png").joinpath("elev_diff_histo.png").exists()


@pytest.mark.parametrize(
    "level",
    [1, 2],
)
def test_run(get_diffanalysis_inputs_config, tmp_path, level):
    """
    Test run function
    """

    user_config = get_diffanalysis_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    workflows = DiffAnalysis(user_config)
    workflows.run()

    assert Path(tmp_path / "csv").joinpath("aligned_elev_stats.csv").exists()

    assert Path(tmp_path / "png").joinpath("diff_elev_after_coreg.png").exists()
    assert Path(tmp_path / "png").joinpath("diff_elev_before_coreg.png").exists()
    assert Path(tmp_path / "png").joinpath("elev_diff_histo.png").exists()
    assert Path(tmp_path / "png").joinpath("masked_elevation.png").exists()
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


@pytest.mark.parametrize(
    "level",
    [1, 2],
)
def test_run_without_coreg(get_diffanalysis_inputs_config, tmp_path, level):
    """
    Test run function
    """

    user_config = get_diffanalysis_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    user_config["coregistration"] = {"process": False}
    workflows = DiffAnalysis(user_config)
    workflows.run()

    assert Path(tmp_path / "csv").joinpath("diff_elev_stats.csv").exists()

    assert Path(tmp_path / "png").joinpath("diff_elev.png").exists()
    assert not Path(tmp_path / "png").joinpath("diff_elev_before_coreg.png").exists()
    assert not Path(tmp_path / "png").joinpath("elev_diff_histo.png").exists()
    assert Path(tmp_path / "png").joinpath("masked_elevation.png").exists()
    assert Path(tmp_path / "png").joinpath("reference_elev_map.png").exists()
    assert Path(tmp_path / "png").joinpath("to_be_aligned_elev_map.png").exists()

    assert not Path(tmp_path / "raster").joinpath("aligned_elev.tif").exists()

    assert Path(tmp_path).joinpath("report.html").exists()
    # sometimes the PDF creation fails for no reason
    # assert Path(tmp_path).joinpath("report.pdf").exists()
    assert Path(tmp_path).joinpath("used_config.yaml").exists()

    csv_files = [
        "diff_elev_stats.csv",
        "reference_elev_stats.csv",
        "to_be_aligned_elev_stats.csv",
    ]

    raster_files = ["diff_elev.tif"]

    if level == 1:
        for file in csv_files:
            assert (Path(tmp_path) / "csv" / file).exists()
        for file in raster_files:
            assert not (Path(tmp_path) / "raster" / file).exists()

    if level == 2:
        for file in csv_files:
            assert (Path(tmp_path) / "csv" / file).exists()
        for file in raster_files:
            assert not (Path(tmp_path) / "raster" / file).exists()


def test_create_html(tmp_path, get_diffanalysis_object_with_run):
    """
    Test create_html function
    """
    _ = get_diffanalysis_object_with_run

    assert Path(tmp_path).joinpath("report.html").exists()


def test_mask_init(tmp_path, get_diffanalysis_inputs_config):
    """
    Test mask initialization
    """
    user_config = get_diffanalysis_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    del user_config["inputs"]["reference_elev"]["path_to_mask"]
    workflows = DiffAnalysis(user_config)
    dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    mask = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
    inlier_mask = ~mask.create_mask(dem)
    assert workflows.inlier_mask == inlier_mask
    assert Path(tmp_path / "png").joinpath("masked_elevation.png").exists()

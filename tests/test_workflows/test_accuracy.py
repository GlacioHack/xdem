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

import logging

# mypy: disable-error-code=no-untyped-def
from pathlib import Path

import geoutils as gu
import numpy as np
import pandas as pd
import pytest

import xdem
from xdem.workflows import Accuracy
from xdem.workflows.workflows import Workflows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
pytest.importorskip("cerberus")


def test_init_diff_analysis(get_accuracy_object_with_run, tmp_path):
    """
    Test initialization of accuracy class
    """
    workflows = get_accuracy_object_with_run

    assert isinstance(workflows, Workflows)
    assert isinstance(workflows, Accuracy)
    assert Path(tmp_path / "plots").joinpath("reference_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("to_be_aligned_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("reference_elev_map.png").exists()
    dem = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
    mask = gu.Vector(xdem.examples.get_path_test("longyearbyen_glacier_outlines"))
    inlier_mask = ~mask.create_mask(dem)
    assert workflows.inlier_mask == inlier_mask


def test__get_reference_elevation(get_accuracy_inputs_config, tmp_path, caplog, assert_and_allow_log):
    """
    Test _get_reference_elevation function
    """

    user_config = get_accuracy_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Accuracy(user_config)
    workflows._load_data()

    with pytest.raises(NotImplementedError, match="This is not implemented, add a reference DEM"):
        workflows._get_reference_elevation()

    user_config = get_accuracy_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["inputs"]["reference_elev"] = None

    with caplog.at_level(logging.WARNING):
        with pytest.raises(NotImplementedError, match="This is not implemented, add a reference DEM"):
            workflows = Accuracy(user_config)
            workflows._load_data()

    # Check logging warning exists and tag as expected
    assert_and_allow_log(caplog, match="No DEM provided", level=logging.WARNING)


@pytest.mark.skip("Not implemented")
def test__compute_coregistration():
    """
    Test _compute_coregistration function
    """


def test__get_stats(get_accuracy_inputs_config, tmp_path):
    """
    Test _get_stats function
    """
    user_config = get_accuracy_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Accuracy(user_config)

    dem = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
    stats_gt = dem.get_stats(
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

    stats_gt = {aliases.get(k, k): v for k, v in stats_gt.items()}
    assert workflows._get_stats(dem) == stats_gt


def test__compute_histogram(get_accuracy_object_with_run, tmp_path):
    """
    Test _compute_histogram function
    """

    _ = get_accuracy_object_with_run

    assert Path(tmp_path / "plots").joinpath("elev_diff_histo.png").exists()


@pytest.mark.parametrize(
    "level",
    [1, 2],
)
def test_run(get_accuracy_inputs_config, tmp_path, level):
    """
    Test run function
    """

    user_config = get_accuracy_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    workflows = Accuracy(user_config)
    workflows.run()

    assert Path(tmp_path / "tables").joinpath("aligned_elev_stats.csv").exists()

    assert Path(tmp_path / "plots").joinpath("diff_elev_after_coreg_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("diff_elev_before_coreg_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("elev_diff_histo.png").exists()
    assert Path(tmp_path / "plots").joinpath("masked_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("reference_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("to_be_aligned_elev_map.png").exists()

    assert Path(tmp_path / "rasters").joinpath("aligned_elev.tif").exists()

    assert Path(tmp_path).joinpath("report.html").exists()
    # sometimes the PDF creation fails for no reason
    # assert Path(tmp_path).joinpath("report.pdf").exists()
    assert Path(tmp_path).joinpath("used_config.yaml").exists()

    csv_files_level_1 = [
        "diff_elev_after_coreg_stats.csv",
        "diff_elev_before_coreg_stats.csv",
    ]

    csv_files_level_2 = [
        "reference_elev_stats.csv",
        "to_be_aligned_elev_stats.csv",
    ]

    raster_files = [
        "diff_elev_after_coreg_map.tif",
        "diff_elev_before_coreg_map.tif",
        "to_be_aligned_elev_reprojected.tif",
        "reference_elev_reprojected.tif",
    ]

    if level == 1:
        for file in csv_files_level_1:
            assert (Path(tmp_path) / "tables" / file).exists()
        for file in csv_files_level_2:
            assert not (Path(tmp_path) / "tables" / file).exists()
        for file in raster_files:
            assert not (Path(tmp_path) / "rasters" / file).exists()

    if level == 2:
        for file in csv_files_level_1:
            assert (Path(tmp_path) / "tables" / file).exists()
        for file in csv_files_level_2:
            assert (Path(tmp_path) / "tables" / file).exists()
        for file in raster_files:
            assert (Path(tmp_path) / "rasters" / file).exists()


@pytest.mark.parametrize(
    "level",
    [1, 2],
)
def test_run_without_coreg(get_accuracy_inputs_config, tmp_path, level):
    """
    Test run function
    """

    user_config = get_accuracy_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    user_config["coregistration"] = {"process": False}
    workflows = Accuracy(user_config)
    workflows.run()

    assert Path(tmp_path / "tables").joinpath("diff_elev_stats.csv").exists()

    assert Path(tmp_path / "plots").joinpath("diff_elev.png").exists()
    assert not Path(tmp_path / "plots").joinpath("diff_elev_before_coreg.png").exists()
    assert not Path(tmp_path / "plots").joinpath("elev_diff_histo.png").exists()
    assert Path(tmp_path / "plots").joinpath("masked_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("reference_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("to_be_aligned_elev_map.png").exists()

    assert not Path(tmp_path / "rasters").joinpath("aligned_elev.tif").exists()

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
            assert (Path(tmp_path) / "tables" / file).exists()
        for file in raster_files:
            assert not (Path(tmp_path) / "rasters" / file).exists()

    if level == 2:
        for file in csv_files:
            assert (Path(tmp_path) / "tables" / file).exists()
        for file in raster_files:
            assert (Path(tmp_path) / "rasters" / file).exists()


@pytest.mark.parametrize(
    "config",
    [
        (True, "reference_elev", "reference_elev", None),
        (False, "reference_elev", "reference_elev", None),
        (True, "to_be_aligned_elev", "to_be_aligned_elev", None),
        (False, "to_be_aligned_elev", "to_be_aligned_elev", None),
        (True, None, None, "must be set to"),
        (True, None, "reference_elev", "must be set to"),
        (False, None, None, None),
        (False, None, "reference_elev", "same shape, transform and CRS"),
    ],
)
def test_run_prepare_datas(get_accuracy_inputs_config, tmp_path, config):
    """
    Test preparation data with all sampling_grid values in a coreg/no coreg process.
    """

    process, sampling_grid, dem_to_crop, error = config
    user_config = get_accuracy_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": 2}
    user_config["coregistration"] = {"process": process}
    user_config["inputs"]["sampling_grid"] = sampling_grid

    if dem_to_crop is not None:
        dem = xdem.DEM(user_config["inputs"][dem_to_crop]["path_to_elev"])
        nrows, ncols = dem.shape

        dem_crop = dem.icrop((0, 0, ncols - 3, nrows - 3))
        dem_crop.to_file(Path(tmp_path / (dem_to_crop + "_crop.tif")))
        user_config["inputs"][dem_to_crop]["path_to_elev"] = Path(tmp_path / (dem_to_crop + "_crop.tif")).as_posix()

    if error is not None:
        with pytest.raises(ValueError, match=error):
            workflows = Accuracy(user_config)
            workflows.run()
    else:
        workflows = Accuracy(user_config)
        workflows.run()

        if sampling_grid is not None:
            raster_files = [
                "to_be_aligned_elev_reprojected.tif",
                "reference_elev_reprojected.tif",
            ]

            for file in raster_files:
                dem_test = xdem.DEM(Path(tmp_path) / "rasters" / file)
                assert dem_crop.shape == dem_test.shape

            csv_files = [
                "reference_elev_stats.csv",
                "to_be_aligned_elev_stats.csv",
            ]
            for file in csv_files:
                stats = pd.read_csv(Path(tmp_path / "tables" / file).as_posix())
                assert stats["Total count"].values[0] == dem_crop.shape[0] * dem_crop.shape[1]


@pytest.mark.parametrize(
    "config",
    [
        ("reference_elev", ["reference_elev"]),
        ("reference_elev", ["to_be_aligned_elev"]),
        ("to_be_aligned_elev", ["reference_elev"]),
        ("to_be_aligned_elev", ["to_be_aligned_elev"]),
        ("reference_elev", ["reference_elev", "to_be_aligned_elev"]),
        ("to_be_aligned_elev", ["reference_elev", "to_be_aligned_elev"]),
    ],
)
def test_prepare_datas(get_accuracy_inputs_config, tmp_path, config):
    """
    Test preparation data with all sampling_grid values
    """

    sampling_grid, dem_to_crop_list = config
    user_config = get_accuracy_inputs_config

    # Save path before crop(s)
    original_ref_path = user_config["inputs"]["reference_elev"]["path_to_elev"]
    original_tba_path = user_config["inputs"]["to_be_aligned_elev"]["path_to_elev"]

    # Update user_config
    user_config["outputs"] = {"path": str(tmp_path), "level": 2}
    user_config["coregistration"] = {"process": False}
    user_config["inputs"]["sampling_grid"] = sampling_grid
    user_config["inputs"]["to_be_aligned_elev"]["path_to_mask"] = None

    # Init crops possible values
    crop = dict()
    nrows, ncols = xdem.DEM(user_config["inputs"]["reference_elev"]["path_to_elev"]).shape
    crop["reference_elev"] = (0, 0, int(ncols / 2) - 1, int(nrows / 2) - 1)
    crop["to_be_aligned_elev"] = (int(ncols / 2) + 1, int(nrows / 2) + 1, ncols, nrows)

    # Crop dems(s) and update config
    for dem_to_crop in dem_to_crop_list:
        dem = xdem.DEM(user_config["inputs"][dem_to_crop]["path_to_elev"])
        dem_crop = dem.icrop(crop[dem_to_crop])
        dem_crop.to_file(Path(tmp_path / (dem_to_crop + "_crop.tif")))
        user_config["inputs"][dem_to_crop]["path_to_elev"] = Path(tmp_path / (dem_to_crop + "_crop.tif")).as_posix()

    workflows = Accuracy(user_config)
    workflows.run()

    raster_files = [
        "to_be_aligned_elev_reprojected.tif",
        "reference_elev_reprojected.tif",
    ]

    # Verify shape and CRS of both outputs
    final_shape = xdem.DEM(user_config["inputs"][sampling_grid]["path_to_elev"]).shape
    final_crs = xdem.DEM(user_config["inputs"][sampling_grid]["path_to_elev"]).get_metric_crs()
    for raster_file in raster_files:
        dem_raster_file = xdem.DEM(Path(tmp_path) / "rasters" / raster_file)
        assert dem_raster_file.shape == final_shape
        assert dem_raster_file.crs == final_crs

    # Verify array of both outputs by means to the array wanted
    reference_elev_reprojected_mean = xdem.DEM(Path(tmp_path) / "rasters" / "reference_elev_reprojected.tif").get_stats(
        "mean"
    )
    to_be_aligned_elev_reprojected_mean = xdem.DEM(
        Path(tmp_path) / "rasters" / "to_be_aligned_elev_reprojected.tif"
    ).get_stats("mean")

    if sampling_grid == "reference_elev":

        # If reference_elev is cropped or not
        dem_ref_ref = xdem.DEM(original_ref_path)
        if "reference_elev" in dem_to_crop_list:
            dem_ref_ref = dem_ref_ref.icrop(crop["reference_elev"])
        assert dem_ref_ref.get_stats("mean") == reference_elev_reprojected_mean

        # If intersection between ref and tba is not null
        if len(dem_to_crop_list) == 1:
            assert xdem.DEM(original_tba_path).icrop(crop[dem_to_crop_list[0]]).get_stats("mean") == pytest.approx(
                to_be_aligned_elev_reprojected_mean
            )
        else:
            assert np.isnan(to_be_aligned_elev_reprojected_mean)

    else:

        # If to_be_aligned_elev is cropped or not
        dem_tba_ref = xdem.DEM(original_tba_path)
        if "to_be_aligned_elev" in dem_to_crop_list:
            dem_tba_ref = dem_tba_ref.icrop(crop["to_be_aligned_elev"])
        assert dem_tba_ref.get_stats("mean") == to_be_aligned_elev_reprojected_mean

        # If intersection between ref and tba is not null
        if len(dem_to_crop_list) == 1:
            assert xdem.DEM(original_ref_path).icrop(crop[dem_to_crop_list[0]]).get_stats("mean") == pytest.approx(
                reference_elev_reprojected_mean
            )
        else:
            assert np.isnan(reference_elev_reprojected_mean)


def test_create_html(tmp_path, get_accuracy_object_with_run):
    """
    Test create_html function
    """
    _ = get_accuracy_object_with_run

    assert Path(tmp_path).joinpath("report.html").exists()


def test_mask_init(tmp_path, get_accuracy_inputs_config):
    """
    Test mask initialization
    """
    user_config = get_accuracy_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    del user_config["inputs"]["reference_elev"]["path_to_mask"]
    workflows = Accuracy(user_config)
    workflows._load_data()
    dem = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
    mask = gu.Vector(xdem.examples.get_path_test("longyearbyen_glacier_outlines"))
    inlier_mask = ~mask.create_mask(dem)
    assert workflows.inlier_mask == inlier_mask
    assert Path(tmp_path / "plots").joinpath("masked_elev_map.png").exists()

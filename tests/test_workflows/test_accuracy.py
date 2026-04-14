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
from xdem.workflows.schemas import MIN_STATS
from xdem.workflows.workflows import _ALIAS

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
pytest.importorskip("cerberus")


def test_init_diff_analysis(get_accuracy_inputs_test):
    """
    Test initialization of accuracy class
    """
    workflows = Accuracy(get_accuracy_inputs_test)
    workflows.run()

    dem = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
    mask = gu.Vector(xdem.examples.get_path_test("longyearbyen_glacier_outlines"))
    inlier_mask = ~mask.create_mask(dem)
    assert workflows.to_be_aligned_elev.get_mask() == inlier_mask


def test__get_reference_elevation(get_accuracy_inputs_test, tmp_path, caplog, assert_and_allow_log):
    """
    Test _get_reference_elevation function
    """

    user_config = get_accuracy_inputs_test
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Accuracy(user_config)
    workflows._load_data()

    with pytest.raises(NotImplementedError, match="This is not implemented, add a reference elevation"):
        workflows._get_reference_elevation()

    user_config = get_accuracy_inputs_test
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["inputs"]["reference_elev"] = None

    with caplog.at_level(logging.WARNING):
        with pytest.raises(NotImplementedError, match="This is not implemented, add a reference elevation"):
            workflows = Accuracy(user_config)
            workflows._load_data()

    # Check logging warning exists and tag as expected
    assert_and_allow_log(caplog, match="No DEM provided", level=logging.WARNING)


@pytest.mark.skip("Not implemented")
def test__compute_coregistration():
    """
    Test _compute_coregistration function
    """


@pytest.mark.parametrize(
    "stats_name, res",
    [
        [MIN_STATS, [_ALIAS.get(k) for k in MIN_STATS]],
        [list(_ALIAS.keys()), [_ALIAS.get(k) for k in _ALIAS.keys()]],
        [["std"], ["Standard deviation"]],
        [["standarddeviation"], ["Standard deviation"]],
        [["std", "standarddeviation"], ["Standard deviation"]],
    ],
)
def test__get_stats(get_accuracy_inputs_test, tmp_path, stats_name, res):
    """
    Test _get_stats function
    """

    user_config = get_accuracy_inputs_test
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["statistics"] = stats_name
    workflows = Accuracy(user_config)

    dem = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
    stats_gt = dem.get_stats(stats_name)

    assert list(set(workflows._get_stats(dem).keys())) == list(set(res))  # type: ignore
    assert workflows._get_stats(dem) == {_ALIAS.get(k, k): v for k, v in stats_gt.items()}


@pytest.mark.parametrize(
    "level",
    [1, 2],
)
def test_run(get_accuracy_inputs_test, tmp_path, level):
    """
    Test run function with (process = True)
    """

    user_config = get_accuracy_inputs_test
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    workflows = Accuracy(user_config)
    workflows.run()

    assert Path(tmp_path / "tables").joinpath("aligned_elev_stats.csv").exists()

    assert Path(tmp_path / "plots").joinpath("diff_elev_diff_coreg_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("elev_diff_histo.png").exists()
    assert Path(tmp_path / "plots").joinpath("masked_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("inputs.png").exists()

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
def test_run_without_coreg(get_accuracy_inputs_test, tmp_path, level):
    """
    Test run function with (process = False)
    """

    user_config = get_accuracy_inputs_test
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    user_config["coregistration"] = {"process": False}
    workflows = Accuracy(user_config)
    workflows.run()

    assert Path(tmp_path / "tables").joinpath("diff_elev_without_coreg_stats.csv").exists()

    assert Path(tmp_path / "plots").joinpath("diff_elev_without_coreg_map.png").exists()
    assert not Path(tmp_path / "plots").joinpath("diff_elev_diff_coreg_map.png").exists()
    assert not Path(tmp_path / "plots").joinpath("elev_diff_histo.png").exists()
    assert Path(tmp_path / "plots").joinpath("masked_elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("inputs.png").exists()

    assert not Path(tmp_path / "rasters").joinpath("aligned_elev.tif").exists()

    assert Path(tmp_path).joinpath("report.html").exists()
    # sometimes the PDF creation fails for no reason
    # assert Path(tmp_path).joinpath("report.pdf").exists()
    assert Path(tmp_path).joinpath("used_config.yaml").exists()

    csv_files = [
        "diff_elev_without_coreg_stats.csv",
        "reference_elev_stats.csv",
        "to_be_aligned_elev_stats.csv",
    ]

    raster_files = ["diff_elev_without_coreg_map.tif"]

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
def test_run_prepare_datas(get_accuracy_inputs_test, tmp_path, config):
    """
    Test preparation data with all sampling_grid values in a coreg/no coreg process.
    """

    process, sampling_grid, dem_to_crop, error = config
    user_config = get_accuracy_inputs_test
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
def test_prepare_datas(get_accuracy_inputs_test, tmp_path, config):
    """
    Test preparation data with all sampling_grid values
    """

    sampling_grid, dem_to_crop_list = config
    user_config = get_accuracy_inputs_test

    # Save path before crop(s)
    original_ref_path = user_config["inputs"]["reference_elev"]["path_to_elev"]
    original_tba_path = user_config["inputs"]["to_be_aligned_elev"]["path_to_elev"]
    user_config["inputs"]["reference_elev"]["path_to_mask"] = None
    user_config["inputs"]["to_be_aligned_elev"]["path_to_mask"] = None

    # Update user_config
    user_config["outputs"] = {"path": str(tmp_path), "level": 2}
    user_config["coregistration"] = {"process": False}
    user_config["inputs"]["sampling_grid"] = sampling_grid
    user_config["inputs"]["reference_elev"]["path_to_mask"] = None
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


@pytest.mark.parametrize(
    "masked",
    [
        [True, True],
        [False, True],
        [True, False],
        [False, False],
    ],
)
def test_mask(tmp_path, get_accuracy_inputs_test, masked):
    """
    Test mask initialization and correg
    """
    user_config = get_accuracy_inputs_test
    masked_ref, masked_tba = masked
    user_config["outputs"] = {"path": str(tmp_path), "level": 2}
    ref_dem_path = xdem.examples.get_path_test("longyearbyen_ref_dem")
    tba_dem_path = xdem.examples.get_path_test("longyearbyen_tba_dem")
    mask_ref_dem_path = xdem.examples.get_path_test("longyearbyen_glacier_outlines")
    mask_tba_dem_path = xdem.examples.get_path_test("longyearbyen_glacier_outlines_2010")

    # Create 1/2 mask (up) for ref and 1/2 mask (bottom) for tba
    ref_dem = xdem.DEM(ref_dem_path)
    ref_dem.load()
    tba_dem = xdem.DEM(tba_dem_path)
    tba_dem.load()
    ref_mask = gu.Vector(mask_ref_dem_path)
    tba_mask = gu.Vector(mask_tba_dem_path)

    user_config["inputs"]["reference_elev"]["path_to_elev"] = ref_dem_path
    if masked_ref:
        inlier_mask = ~ref_mask.create_mask(ref_dem)
        inlier_mask_reproject = inlier_mask.reproject(ref_dem).crop(ref_dem)
        ref_dem.set_mask(~inlier_mask_reproject)
        user_config["inputs"]["reference_elev"]["path_to_mask"] = mask_ref_dem_path
    else:
        user_config["inputs"]["reference_elev"]["path_to_mask"] = None

    user_config["inputs"]["to_be_aligned_elev"]["path_to_elev"] = tba_dem_path
    if masked_tba:
        inlier_mask = ~tba_mask.create_mask(tba_dem)
        inlier_mask_reproject = inlier_mask.reproject(tba_dem).crop(tba_dem)
        tba_dem.set_mask(~inlier_mask_reproject)
        user_config["inputs"]["to_be_aligned_elev"]["path_to_mask"] = mask_tba_dem_path
    else:
        user_config["inputs"]["to_be_aligned_elev"]["path_to_mask"] = None

    # Apply to config dict
    workflows = Accuracy(user_config)
    workflows.run()

    # Verify 1/2 mask application for ref data
    stats_ref = pd.read_csv(Path(tmp_path / "tables" / "reference_elev_stats.csv").as_posix())
    assert stats_ref["Valid count"].values[0] == ref_dem.get_stats("Valid count")

    # Count 1/2 mask application for tba data
    stats_tba = pd.read_csv(Path(tmp_path / "tables" / "to_be_aligned_elev_stats.csv").as_posix())
    assert stats_tba["Valid count"].values[0] == tba_dem.get_stats("Valid count")

    stats_tba_aligned = pd.read_csv(Path(tmp_path / "tables" / "aligned_elev_stats.csv").as_posix())
    aligned_tba = tba_dem.coregister_3d(ref_dem, xdem.coreg.LZD(subsample=10000), random_state=42)
    assert stats_tba_aligned["Valid count"].values[0] == aligned_tba.get_stats("Valid count")

    # Count full mask on diff elev data
    stats_before = pd.read_csv(Path(tmp_path / "tables" / "diff_elev_before_coreg_stats.csv").as_posix())
    stats_after = pd.read_csv(Path(tmp_path / "tables" / "diff_elev_after_coreg_stats.csv").as_posix())

    diff_before = tba_dem - ref_dem
    assert stats_before["Valid count"].values[0] == diff_before.get_stats("Valid count")
    diff_after = aligned_tba.reproject(ref_dem) - ref_dem
    assert stats_after["Valid count"].values[0] == diff_after.get_stats("Valid count")


@pytest.mark.skip("Todo when VCRS will be a part of CRS")
@pytest.mark.parametrize("vcrs_first_step", [[None, "Ellipsoid"], ["Ellipsoid", None]])
# @pytest.mark.parametrize("vcrs_second_step", [[None, "EGM96"],["EGM96", None]])
@pytest.mark.parametrize("sampling_grid_first_step", ["reference_elev", "to_be_aligned_elev"])
# @pytest.mark.parametrize("sampling_grid_second_step", ["reference_elev", "to_be_aligned_elev"])
def test_vcrs_change(
    tmp_path, get_accuracy_inputs_test, vcrs_first_step, sampling_grid_first_step  # vcrs_second_step,
):
    user_config = get_accuracy_inputs_test
    user_config["inputs"]["reference_elev"]["set_vcrs"] = vcrs_first_step[0]
    user_config["inputs"]["to_be_aligned_elev"]["set_vcrs"] = vcrs_first_step[1]

    ref = xdem.DEM(user_config["inputs"]["reference_elev"]["path_to_elev"])
    ref.set_vcrs("Ellipsoid")
    ref.to_vcrs("EGM96", inplace=True)

    user_config["inputs"]["reference_elev"]["path_to_mask"] = None
    user_config["inputs"]["sampling_grid"] = sampling_grid_first_step
    user_config["outputs"] = {"path": str(tmp_path), "level": 2}
    workflows = Accuracy(user_config)
    workflows.run()

    if sampling_grid_first_step == "reference_elev":
        vcrs_res = vcrs_first_step[0]
    else:
        vcrs_res = vcrs_first_step[1]

    assert xdem.DEM(Path(tmp_path / "rasters" / "reference_elev_reprojected.tif")).vcrs == vcrs_res
    assert xdem.DEM(Path(tmp_path / "rasters" / "to_be_aligned_elev_reprojected.tif")).vcrs == vcrs_res
    assert xdem.DEM(Path(tmp_path / "rasters" / "aligned_elev.tif")).vcrs == vcrs_res

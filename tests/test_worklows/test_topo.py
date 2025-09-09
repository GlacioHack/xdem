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
Test Topo class
"""
# mypy: disable-error-code=no-untyped-def
from pathlib import Path

import pytest
from rasterio import Affine

import xdem
from xdem.workflows import Topo
from xdem.workflows.workflows import Workflows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_init_topo_summary(get_topo_inputs_config, tmp_path, list_default_terrain_attributes):
    """
    Test Topo class initialization
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)

    assert isinstance(workflows, Workflows)
    assert isinstance(workflows, Topo)
    assert Path(tmp_path / "plots").joinpath("elevation (m).png").exists()
    assert Path(tmp_path / "plots").joinpath("masked_elevation.png").exists()
    assert workflows.config_attributes == list_default_terrain_attributes

    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["terrain_attributes"] = []
    workflows = Topo(user_config)
    assert workflows.config_attributes == []

    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["terrain_attributes"] = {
        "hillshade": {"extra_information": {"method": "ZevenbergThorne", "azimuth": 90}}
    }
    workflows = Topo(user_config)
    assert workflows.config_attributes == {
        "hillshade": {"extra_information": {"method": "ZevenbergThorne", "azimuth": 90}}
    }
    assert workflows.list_attributes == ["hillshade"]


def test_generate_terrain_attributes(tmp_path, get_topo_inputs_config, list_default_terrain_attributes):
    """
    Test generate_terrain_attributes function
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)

    workflows.generate_terrain_attributes_tiff()

    for attr in list_default_terrain_attributes:
        assert Path(tmp_path / "rasters").joinpath(f"{attr}.tif").exists()


def test_generate_terrain_attributes_level_2(tmp_path, get_topo_inputs_config, list_default_terrain_attributes):
    """
    Test generate_terrain_attributes function
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": 2}
    workflows = Topo(user_config)
    workflows.run()

    for attr in list_default_terrain_attributes:
        assert Path(tmp_path / "rasters").joinpath(f"{attr}.tif").exists()


def test_generate_no_terrain_attributes(tmp_path, get_topo_inputs_config, list_default_terrain_attributes):
    """
    Test generate_terrain_attributes function
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path), "level": 2}
    user_config["terrain_attributes"] = None
    workflows = Topo(user_config)
    workflows.run()

    for attr in list_default_terrain_attributes:
        assert not Path(tmp_path / "rasters").joinpath(f"{attr}.tif").exists()


def test_generate_terrain_attributes_png(tmp_path, get_topo_inputs_config):
    """
    Test generate_terrain_attributes_png function
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    workflows.run()

    workflows.generate_terrain_attributes_png()
    assert Path(tmp_path / "plots").joinpath("terrain_attributes.png").exists()


def test_run(get_topo_inputs_config, tmp_path):
    """
    Test run function
    """

    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    workflows.run()

    assert Path(tmp_path / "tables").joinpath("stats_elev_stats.csv").exists()
    assert Path(tmp_path / "tables").joinpath("stats_elev_mask_stats.csv").exists()
    assert Path(tmp_path).joinpath("report.html").exists()
    assert workflows.dico_to_show == [
        (
            "Information about inputs",
            {
                "reference_elev": {
                    "path_to_elev": xdem.examples.get_path("longyearbyen_tba_dem"),
                    "from_vcrs": "EGM96",
                    "path_to_mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
                    "to_vcrs": "EGM96",
                }
            },
        ),
        (
            "DEM information",
            {
                "Data types": "float32",
                "Driver": "GTiff",
                "Filename": xdem.examples.get_path("longyearbyen_tba_dem"),
                "Grid size": "us_nga_egm96_15.tif",
                "Height": 985,
                "Nodata Value": -9999.0,
                "Number of band": (1,),
                "Pixel interpretation": "Area",
                "Pixel size": (20.0, 20.0),
                "Transform": Affine(20.0, 0.0, 502810.0, 0.0, -20.0, 8674030.0),
                "Width": 1332,
            },
        ),
        (
            "Global statistics",
            {
                "90thpercentile": 727.55,
                "le90": 766.79,
                "max": 1022.29,
                "mean": 381.32,
                "median": 365.23,
                "min": 8.38,
                "nmad": 291.27,
                "percentagevalidpoints": 100.0,
                "rmse": 452.68,
                "standarddeviation": 243.95,
                "std": 243.95,
                "sum": 500301600.0,
                "sumofsquares": 268858540032.0,
                "totalcount": 1312020,
                "validcount": 1312020,
            },
        ),
        (
            "Mask statistics",
            {
                "90thpercentile": 788.29,
                "le90": 498.87,
                "max": 1011.9,
                "mean": 592.41,
                "median": 591.89,
                "min": 139.9,
                "nmad": 163.87,
                "percentagevalidpoints": 100.0,
                "rmse": 611.76,
                "standarddeviation": 152.66,
                "std": 152.66,
                "sum": 98813536.0,
                "sumofsquares": 62425526272.0,
                "totalcount": 1312020,
                "validcount": 1312020,
            },
        ),
    ]

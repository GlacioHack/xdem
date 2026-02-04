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
from rasterio.coords import BoundingBox

import xdem
from xdem.workflows import Topo
from xdem.workflows.schemas import STATS_METHODS
from xdem.workflows.workflows import Workflows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

pytest.importorskip("cerberus")


def test_init_topo_summary(get_topo_inputs_config, tmp_path, list_default_terrain_attributes):
    """
    Test Topo class initialization
    """
    user_config = get_topo_inputs_config
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    workflows._load_data()

    assert isinstance(workflows, Workflows)
    assert isinstance(workflows, Topo)
    assert Path(tmp_path / "plots").joinpath("elev_map.png").exists()
    assert Path(tmp_path / "plots").joinpath("masked_elev_map.png").exists()
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
    workflows._load_data()
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
    workflows._load_data()
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
    assert Path(tmp_path / "plots").joinpath("terrain_attributes_map.png").exists()


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
    # Check subdictionaries content, except exact stats values in case test data/algorithms slightly changes,
    # and as those are already tested separately
    # 1/ Input information
    assert workflows.dico_to_show[0] == (
        "Information about inputs",
        {
            "reference_elev": {
                "path_to_elev": xdem.examples.get_path_test("longyearbyen_tba_dem"),
                "from_vcrs": None,
                "path_to_mask": xdem.examples.get_path_test("longyearbyen_glacier_outlines"),
                "to_vcrs": None,
                "downsample": 1,
            }
        },
    )
    # 2/ Elevation information
    assert workflows.dico_to_show[1] == (
        "Elevation information",
        {
            "Data types": "float32",
            "Driver": "GTiff",
            "Filename": xdem.examples.get_path_test("longyearbyen_tba_dem"),
            "Grid size": None,
            "Height": 54,
            "Nodata Value": -9999.0,
            "Number of band": (1,),
            "Pixel interpretation": "Area",
            "Pixel size": (20.0, 20.0),
            "Transform": Affine(20.0, 0.0, 512310.0, 0.0, -20.0, 8662030.0),
            "Width": 70,
            "Bounds": BoundingBox(left=512310.0, bottom=8660950.0, right=513710.0, top=8662030.0),
        },
    )

    # 3/ Statistics names
    assert workflows.dico_to_show[2][0] == "Global statistics"
    assert list(workflows.dico_to_show[2][1].keys()) == STATS_METHODS
    assert workflows.dico_to_show[3][0] == "Mask statistics"
    assert list(workflows.dico_to_show[3][1].keys()) == STATS_METHODS

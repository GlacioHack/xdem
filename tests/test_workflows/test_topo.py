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

from collections import OrderedDict

# mypy: disable-error-code=no-untyped-def
from pathlib import Path

import geoutils as gu
import pytest
from rasterio import Affine
from rasterio.coords import BoundingBox

import xdem
from xdem.workflows import Topo
from xdem.workflows.schemas import MIN_STATS, TERRAIN_ATTRIBUTES_DEFAULT
from xdem.workflows.workflows import _ALIAS

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

pytest.importorskip("cerberus")


def files_load_data(input, tmp_path):
    assert Path(tmp_path / "plots").joinpath("elev_map.png").exists()
    if "path_to_mask" in input:
        assert Path(tmp_path / "plots").joinpath("masked_elev_map.png").exists()


def files_attribute(workflows, level, attributes, tmp_path):
    if isinstance(attributes, list):
        for attr in attributes:
            if attributes:
                assert Path(tmp_path / "plots").joinpath("terrain_attributes_map.png").exists()
                if level == 1:
                    assert not Path(tmp_path / "rasters").joinpath(f"{attr}.tif").exists()
                else:
                    assert Path(tmp_path / "rasters").joinpath(f"{attr}.tif").exists()
    else:
        assert workflows.list_attributes == list(attributes.keys())
        attribute = list(attributes.keys())[0]

        assert Path(tmp_path / "plots").joinpath("terrain_attributes_map.png").exists()
        if level == 1:
            assert not Path(tmp_path / "rasters").joinpath(f"{attribute}.tif").exists()
        else:
            assert Path(tmp_path / "rasters").joinpath(f"{attribute}.tif").exists()


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("nb_inputs", [1, 2])
@pytest.mark.parametrize(
    "attributes",
    [None, ["slope", "aspect"], {"hillshade": {"extra_information": {"method": "ZevenbergThorne", "azimuth": 90}}}],
)
def test_run(tmp_path, level, attributes, nb_inputs, get_topo_inputs_config_list):
    """
    Test run function with all the outputs generation
    """
    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:nb_inputs]
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    user_config["terrain_attributes"] = attributes

    workflows = Topo(user_config)
    workflows.run()

    # 1/ Test inputs
    if nb_inputs == 1:
        files_load_data(user_config["inputs"][0], tmp_path)
    else:
        for k in range(len(user_config["inputs"])):
            dem_dir = "dem_" + str(k)
            files_load_data(user_config["inputs"], Path(tmp_path / dem_dir))

    # 2/ Test attributes
    if attributes is not None:
        if len(attributes) == 0:
            attributes = TERRAIN_ATTRIBUTES_DEFAULT
        assert workflows.config_attributes == attributes
        if nb_inputs == 1:
            files_attribute(workflows, level, attributes, tmp_path)
        else:
            for k in range(len(user_config["inputs"])):
                dem_dir = "dem_" + str(k)
                files_attribute(workflows, level, attributes, Path(tmp_path / dem_dir))
    else:
        assert workflows.config_attributes is None
        assert not Path(tmp_path / "plots").joinpath("terrain_attributes_map.png").exists()

    # 3/ Test stats
    if nb_inputs == 1:
        assert Path(tmp_path / "tables").joinpath("stats_elev_stats.csv").exists()
    else:
        for k in range(len(user_config["inputs"])):
            dem_dir = "dem_" + str(k)
            assert Path(tmp_path / dem_dir / "tables").joinpath("stats_elev_stats.csv").exists()


@pytest.mark.parametrize("nb_inputs", [1, 2])
def test_run_dico_to_show(get_topo_inputs_config_list, nb_inputs, tmp_path):
    """
    Test run function and dico_to_show values
    """

    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:nb_inputs]
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    workflows.run()

    # Check subdictionaries content, except exact stats values in case test data/algorithms slightly changes,
    # and as those are already tested separately

    for k, _ in enumerate(user_config["inputs"]):

        # 1/ Input information
        get_topo_inputs_config_list[k]["downsample"] = 1
        assert workflows.dico_to_show[k][0] == (
            "Information about inputs",
            get_topo_inputs_config_list[k],
        )

        # 2/ Elevation information
        assert workflows.dico_to_show[k][1] == (
            "Elevation information",
            {
                "Data types": "float32",
                "Driver": "GTiff",
                "Filename": get_topo_inputs_config_list[k]["path_to_elev"],
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
        assert workflows.dico_to_show[k][2][0] == "Statistics"
        assert list(workflows.dico_to_show[k][2][1].keys()) == [_ALIAS.get(k) for k in MIN_STATS]

        dem = xdem.DEM(user_config["inputs"][k]["path_to_elev"])
        if "path_to_mask" in user_config["inputs"][k]:
            ref_mask = gu.Vector(user_config["inputs"][k]["path_to_mask"])
            dem.load()
            inlier_mask = ~ref_mask.create_mask(dem)
            dem.set_mask(~inlier_mask)
        res = workflows.floats_process(dem.get_stats(MIN_STATS))
        assert workflows.dico_to_show[k][2][1] == {_ALIAS.get(key): res[key] for key in res.keys()}


@pytest.mark.parametrize("nb_inputs", [1, 2])
@pytest.mark.parametrize(
    "stats_name, res",
    [
        [MIN_STATS, [_ALIAS.get(k) for k in MIN_STATS]],
        [list(_ALIAS.keys()), list(OrderedDict((x, True) for x in [_ALIAS.get(k) for k in _ALIAS.keys()]).keys())],
        [["std"], ["Standard deviation"]],
        [["standarddeviation"], ["Standard deviation"]],
        [["std", "standarddeviation"], ["Standard deviation"]],
    ],
)
def test_stats_list(get_topo_inputs_config_list, nb_inputs, tmp_path, stats_name, res):
    """
    Test to check the output stats from several input stats list
    """

    user_config = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:nb_inputs]
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["statistics"] = stats_name
    workflows = Topo(user_config)
    workflows.run()

    for k, _ in enumerate(user_config["inputs"]):
        assert list(workflows.dico_to_show[k][2][1].keys()) == res

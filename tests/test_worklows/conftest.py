from typing import Any, Dict

import pytest

import xdem


@pytest.fixture()
def get_topo_inputs_config() -> Dict[str, Any]:
    """ """
    return {
        "inputs": {
            "dem": xdem.examples.get_path("longyearbyen_tba_dem"),
            "mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
        },
    }


@pytest.fixture()
def get_compare_inputs_config() -> Dict[str, Any]:
    """ """
    return {
        "inputs": {
            "reference_elev": {
                "dem": xdem.examples.get_path("longyearbyen_tba_dem"),
                "mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            },
            "to_be_aligned_elev": {
                "dem": xdem.examples.get_path("longyearbyen_tba_dem"),
                "mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            },
        },
    }


@pytest.fixture()
def pipeline_topo():
    """ """
    return {
        "inputs": {
            "dem": xdem.examples.get_path("longyearbyen_tba_dem"),
            "mask": xdem.examples.get_path("longyearbyen_glacier_outlines"),
            "from_vcrs": {"common": "EGM96"},
            "to_vcrs": {"common": "EGM96"},
        },
        "statistics": [
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
        ],
        "terrain_attributes": ["hillshade", "slope", "aspect", "curvature", "terrain_ruggedness_index", "rugosity"],
        "outputs": {"path": "outputs", "level": 1},
    }

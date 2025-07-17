from typing import Dict, Any

import pytest

import xdem


@pytest.fixture()
def get_info_inputs_config() -> Dict[str, Any]:
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

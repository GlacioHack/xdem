"""
DEM coregistration classes and functions, including affine methods, bias corrections (i.e. non-affine) and filters.
"""

from xdem.coreg.affine import (  # noqa
    ICP,
    GradientDescending,
    NuthKaab,
    Tilt,
    VerticalShift,
)
from xdem.coreg.base import BlockwiseCoreg, Coreg, CoregPipeline  # noqa
from xdem.coreg.biascorr import (  # noqa
    BiasCorr1D,
    BiasCorr2D,
    BiasCorrND,
    Deramp,
    DirectionalBias,
    TerrainBias,
)
from xdem.coreg.pipelines import dem_coregistration  # noqa

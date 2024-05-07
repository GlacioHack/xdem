"""
DEM coregistration classes and functions, including affine methods, bias corrections (i.e. non-affine) and filters.
"""

from xdem.coreg.affine import (  # noqa
    ICP,
    AffineCoreg,
    GradientDescending,
    NuthKaab,
    Tilt,
    VerticalShift,
)
from xdem.coreg.base import (  # noqa
    BlockwiseCoreg,
    Coreg,
    CoregPipeline,
    apply_matrix,
    invert_matrix,
)
from xdem.coreg.biascorr import BiasCorr, Deramp, DirectionalBias, TerrainBias  # noqa
from xdem.coreg.workflows import dem_coregistration  # noqa

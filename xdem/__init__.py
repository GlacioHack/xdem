from xdem import (  # noqa
    biascorr,
    coreg,
    dem,
    examples,
    filters,
    fit,
    spatialstats,
    terrain,
    volume,
)
from xdem.biascorr import (  # noqa
    BiasCorr,
    BiasCorr1D,
    BiasCorr2D,
    BiasCorrND,
    Deramp,
    DirectionalBias,
    TerrainBias,
)
from xdem.coreg import (  # noqa
    ICP,
    BlockwiseCoreg,
    Coreg,
    CoregPipeline,
    NuthKaab,
    Rigid,
    Tilt,
    VerticalShift,
)
from xdem.ddem import dDEM  # noqa
from xdem.dem import DEM  # noqa
from xdem.demcollection import DEMCollection  # noqa

try:
    from xdem.version import version as __version__  # noqa
except ImportError:  # pragma: no cover
    raise ImportError(
        "xDEM is not properly installed. If you are "
        "running from the source directory, please instead "
        "create a new virtual environment (using conda or "
        "virtualenv) and then install it in-place by running: "
        "pip install -e ."
    )

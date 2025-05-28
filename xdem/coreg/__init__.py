# Copyright (c) 2024 xDEM developers
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
DEM coregistration classes and functions, including affine methods, bias corrections (i.e. non-affine) and filters.
"""

from xdem.coreg.affine import (  # noqa
    CPD,
    ICP,
    LZD,
    AffineCoreg,
    DhMinimize,
    NuthKaab,
    VerticalShift,
)
from xdem.coreg.base import (  # noqa
    Coreg,
    CoregPipeline,
    apply_matrix,
    invert_matrix,
    matrix_from_translations_rotations,
    translations_rotations_from_matrix,
)
from xdem.coreg.biascorr import BiasCorr, Deramp, DirectionalBias, TerrainBias  # noqa
from xdem.coreg.blockwise import BlockwiseCoreg  # noqa

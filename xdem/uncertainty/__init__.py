# Copyright (c) 2025 xDEM developers
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

"""Error structures, estimation and uncertainty propagation workflows."""

from xdem.uncertainty.error_structure import (
    ErrorComponent,
    ErrorMagnitude,
    ErrorStructure,
)
from xdem.uncertainty.numerical import PropagationResult
from xdem.uncertainty.uncertainty import (
    estimate_error_structure,
    number_effective_samples,
    patches_method,
    propagate_uncertainty,
    propagate_uncertainty_coreg,
    spatial_error_propagation,
)

__all__ = [
    "ErrorComponent",
    "ErrorMagnitude",
    "ErrorStructure",
    "PropagationResult",
    "estimate_error_structure",
    "number_effective_samples",
    "patches_method",
    "propagate_uncertainty",
    "propagate_uncertainty_coreg",
    "spatial_error_propagation",
]

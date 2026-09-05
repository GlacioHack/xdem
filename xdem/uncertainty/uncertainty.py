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

"""Public workflows for estimating and propagating elevation uncertainty.

Spatial input preparation is followed by estimation and numerical propagation entry points. Error models, fitting
and analytical or numerical algorithms live in their corresponding uncertainty modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import geopandas as gpd
import numpy as np
from geoutils import PointCloud, Raster, Vector

from xdem import terrain
from xdem._typing import NDArrayb
from xdem.uncertainty.analytical import (
    number_effective_samples,
    spatial_error_propagation,
)
from xdem.uncertainty.error_structure import ErrorStructure
from xdem.uncertainty.numerical import (
    PropagationResult,
    _propagate_uncertainty_coreg,
    _propagate_uncertainty_spatial,
    _propagate_uncertainty_terrain,
    _run_simulations,
    patches_method,
)

__all__ = [
    "estimate_error_structure",
    "propagate_uncertainty",
    "propagate_uncertainty_coreg",
    "number_effective_samples",
    "spatial_error_propagation",
    "patches_method",
]


############################################
# 1/ ESTIMATION FROM ELEVATION DIFFERENCES
############################################


def estimate_error_structure(
    source_elev: Raster | PointCloud | gpd.GeoDataFrame,
    other_elev: Raster | PointCloud | gpd.GeoDataFrame,
    *,
    stable_terrain: Raster | Vector | NDArrayb | gpd.GeoDataFrame | None = None,
    predictors: Mapping[str, Any] | tuple[Any, ...] | None = None,
    components: Mapping[str, Mapping[str, Any]] | None = None,
    other_error: Literal["negligible", "same"] = "negligible",
    z_name: str = "z",
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> ErrorStructure:
    """Estimate an error structure from two elevation datasets on stable terrain.

    Values are compared on their common support through GeoUtils cosampling. A raster grid is retained for two raster
    inputs, while any point cloud defines the support for a mixed comparison. The resulting structure describes the
    source dataset under the selected assumption about the other dataset.

    :param source_elev: Elevation dataset whose error structure is estimated.
    :param other_elev: Comparison elevation dataset with negligible or equal errors.
    :param stable_terrain: Spatial or Boolean mask where elevation differences represent error.
    :param predictors: Named magnitude predictors or an ordered tuple using terrain attribute names where possible.
    :param components: Ordered component specifications passed to :meth:`ErrorStructure.estimate`.
    :param other_error: Whether comparison errors are negligible or have the same structure as the source.
    :param z_name: Elevation column selected from plain GeoDataFrames.
    :param random_state: Random generator or seed used throughout estimation.
    :param kwargs: Additional options passed to :meth:`ErrorStructure.estimate`.
    :returns: Fitted component based error structure.
    """

    # Normalize plain point dataframes to the shared GeoUtils spatial interface
    if isinstance(source_elev, gpd.GeoDataFrame):
        source_elev = PointCloud(source_elev, data_column=z_name)
    if isinstance(other_elev, gpd.GeoDataFrame):
        other_elev = PointCloud(other_elev, data_column=z_name)

    # Require the shared spatial sampling interface before choosing comparison locations
    is_source_spatial = hasattr(source_elev, "cosample")
    is_other_spatial = hasattr(other_elev, "cosample")
    if not is_source_spatial or not is_other_spatial:
        raise TypeError("source_elev and other_elev must be raster or point cloud objects.")

    # Limit attribution to negligible or equal independent errors in the comparison dataset
    if other_error not in {"negligible", "same"}:
        raise ValueError("other_error must be 'negligible' or 'same'.")

    # Prefer source terrain for derived variables and otherwise use the comparison raster
    source_raster = source_elev if hasattr(source_elev, "ij2xy") else None
    other_raster = other_elev if hasattr(other_elev, "ij2xy") else None
    terrain_source = source_raster if source_raster is not None else other_raster

    # Name ordered predictor shorthand and use terrain defaults only when a raster can provide them
    if predictors is None:
        predictor_specs: dict[str, Any] = (
            {"slope": "slope", "max_curvature": "max_curvature"} if terrain_source is not None else {}
        )

    # Keep explicit predictor names, or derive names for the ordered shorthand form
    elif isinstance(predictors, Mapping):
        predictor_specs = dict(predictors)
    else:
        predictor_specs = {
            specification if isinstance(specification, str) else f"predictor_{index + 1}": specification
            for index, specification in enumerate(predictors)
        }

    # Resolve derived variables and aligned source columns before the shared spatial sampling
    resolved_predictors: dict[str, Any] = {}
    auxiliary_at: dict[str, Literal["self", "other"]] = {}
    for name, specification in predictor_specs.items():
        if isinstance(specification, str) and terrain_source is not None and hasattr(terrain, specification):
            resolved_predictors[name] = getattr(terrain, specification)(terrain_source)

        # Read named point attributes in their native order and bind them to the source dataset
        elif isinstance(specification, str) and hasattr(source_elev, "ds") and specification in source_elev.ds:
            resolved_predictors[name] = source_elev.ds[specification].to_numpy()
            auxiliary_at[name] = "self"
        elif isinstance(specification, str):
            raise ValueError(f"Cannot resolve magnitude predictor {specification!r} from the elevation inputs.")
        else:
            resolved_predictors[name] = specification
            if not hasattr(specification, "ij2xy") and not hasattr(specification, "georeferenced_coords_equal"):
                auxiliary_at[name] = "self"

    # Wrap raster shaped masks so mixed point comparisons can evaluate them spatially
    comparison_mask: Any = stable_terrain
    if isinstance(stable_terrain, np.ndarray):
        for raster_support in (source_raster, other_raster):
            if raster_support is not None and stable_terrain.squeeze().shape == raster_support.shape:
                comparison_mask = raster_support.copy(new_array=stable_terrain.squeeze().astype(bool))
                break

    # Evaluate both elevations and predictors only at common stable finite locations
    sample = source_elev.cosample(
        other_elev,
        auxiliary=resolved_predictors,
        auxiliary_at=auxiliary_at or None,
        at="self" if source_raster is not None and other_raster is not None else None,
        mask=comparison_mask,
        subsample=1,
        random_state=random_state,
        align="reproject",
    )

    # Equal independent errors each contribute half of the observed difference variance
    scale = 1 / np.sqrt(2) if other_error == "same" else 1.0

    # Read the common raster bands or point columns as an error proxy on their native support
    if isinstance(sample, Raster):
        proxy_values = scale * (sample.data[0] - sample.data[1])
        error_proxy = source_elev.copy(new_array=proxy_values)
        fitted_predictors: Mapping[str, Any] = {
            name: sample.data[index + 2] for index, name in enumerate(resolved_predictors)
        }
        support_kind = "raster"
    else:
        error_proxy = sample.copy(new_array=scale * (sample.ds["self"] - sample.ds["other"]).to_numpy())
        fitted_predictors = {name: name for name in resolved_predictors}
        support_kind = "pointcloud"

    # Add source attribution metadata after estimating the statistical decomposition
    structure = ErrorStructure.estimate(
        error_proxy,
        predictors=fitted_predictors,
        components=components,
        random_state=random_state,
        **kwargs,
    )

    # Record the attribution assumption and comparison locations with the fitted components
    return ErrorStructure(
        structure.components,
        empirical_variogram=structure.empirical_variogram,
        fit_diagnostics=structure.fit_diagnostics,
        metadata={
            **structure.metadata,
            "other_error": other_error,
            "error_proxy_scale": scale,
            "support": support_kind,
        },
    )


######################################
# 2/ NUMERICAL PROPAGATION WORKFLOWS
######################################


def propagate_uncertainty(
    elev: Raster | PointCloud,
    operation: Callable[[Any], Any] | Literal["spatial", "terrain", "coreg"],
    *,
    error_structure: ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    nsim: int = 100,
    random_state: int | np.random.Generator | None = None,
    return_samples: bool = False,
    **kwargs: Any,
) -> PropagationResult:
    """Propagate elevation uncertainty numerically through a calculation.

    Each realization adds a zero mean Gaussian field with the ErrorStructure's component covariances to the original
    elevations. Predictors of error magnitude stay fixed while the operation is recomputed. Mean and standard
    deviation are accumulated without retaining individual outputs unless requested.

    :param elev: Elevation raster or point cloud to perturb.
    :param operation: Callable accepting one perturbed dataset, or a built-in spatial, terrain or coreg workflow.
    :param error_structure: Fitted or manually defined elevation error structure.
    :param predictors: Magnitude predictors evaluated on the elevation support.
    :param nsim: Number of realizations, at least two.
    :param random_state: Generator or seed for reproducible simulations.
    :param return_samples: Whether to retain individual operation outputs.
    :param kwargs: Simulation options circular_period and on_error; spatial options areas and weights; terrain
        options attribute and terrain_kwargs; coreg options reference_elev, coreg_method, inlier_mask,
        error_applied_to, precoreg and kwargs_coreg_fit.
    :returns: Original estimate, ensemble summaries, valid counts and optional simulation outputs.
    """

    # Keep one set of simulation controls shared by custom and specialized calculations
    options = dict(
        error_structure=error_structure,
        predictors=predictors,
        nsim=nsim,
        random_state=random_state,
        return_samples=return_samples,
        **kwargs,
    )

    # Pass a custom calculation directly to the shared simulation workflow
    if callable(operation):
        return _run_simulations(elev, operation, **options)

    # Resolve the named calculation to the adapter that prepares its fixed inputs
    workflows: dict[str, Callable[..., PropagationResult]] = {
        "spatial": _propagate_uncertainty_spatial,
        "terrain": _propagate_uncertainty_terrain,
        "coreg": _propagate_uncertainty_coreg,
    }

    # Reject unknown workflows before generating any error realizations
    if operation not in workflows:
        raise ValueError("operation must be callable, 'spatial', 'terrain' or 'coreg'.")
    return workflows[operation](elev, **options)


def propagate_uncertainty_coreg(
    reference_elev: Raster | PointCloud,
    to_be_aligned_elev: Raster | PointCloud,
    coreg_method: Any,
    *,
    error_structure: ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    nsim: int = 30,
    error_applied_to: Literal["ref", "tba"] = "tba",
    inlier_mask: Any = None,
    precoreg: bool = False,
    random_state: int | np.random.Generator | None = None,
    kwargs_coreg_fit: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, list[Any]]:
    """Propagate uncertainty to fitted translations and rotations.

    This convenience wrapper retains the coregistration summary table, simulation table and fitted objects. The
    common workflow ``propagate_uncertainty(..., operation="coreg")`` instead returns a PropagationResult.

    :param reference_elev: Reference elevations.
    :param to_be_aligned_elev: Elevations to align to the reference.
    :param coreg_method: Coregistration method copied independently for each fit.
    :param error_structure: Error model of the dataset receiving simulated errors.
    :param predictors: Magnitude predictors on that dataset's support.
    :param nsim: Number of realizations, at least two.
    :param error_applied_to: Whether reference or to-be-aligned elevations receive errors.
    :param inlier_mask: Mask passed to each fit.
    :param precoreg: Whether to align the inputs once before residual simulations.
    :param random_state: Generator or seed used by field generation and fitting.
    :param kwargs_coreg_fit: Additional options passed to Coreg.fit.
    :returns: Mean/STD table, per-simulation metadata and fitted coregistration objects.
    """

    import pandas as pd

    # Use the common propagation workflow so field generation and failure counts stay consistent
    result = propagate_uncertainty(
        to_be_aligned_elev,
        "coreg",
        reference_elev=reference_elev,
        coreg_method=coreg_method,
        error_structure=error_structure,
        predictors=predictors,
        nsim=nsim,
        error_applied_to=error_applied_to,
        inlier_mask=inlier_mask,
        precoreg=precoreg,
        random_state=random_state,
        kwargs_coreg_fit=kwargs_coreg_fit,
    )

    # Format the six parameter summaries and retain the requested and successful simulation counts
    summary = pd.DataFrame({"mean": result.mean, "std": result.std})
    summary.attrs.update(nsim=result.nsim, n_success=result.n_success, frac_success=result.n_success / result.nsim)
    return summary, result.metadata["simulation_table"], result.metadata["fitted_coregs"]

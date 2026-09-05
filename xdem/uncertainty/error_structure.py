# Copyright (c) 2026 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Error magnitudes, spatial correlation components and their combined structure.

An error structure is represented as a sum of named independent components. Each component combines an error
magnitude with a correlation model whose variance is one, which keeps the amplitude and spatial scale unambiguous.

The module defines magnitude models first, followed by individual components and their combined structure. Prediction,
covariance conversion, plotting and random field generation are exposed on these compact public objects, while fitting
from an error proxy is implemented separately in :mod:`xdem.uncertainty.estimation`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import geoutils as gu
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist

from xdem.fit import interp_binning

if TYPE_CHECKING:
    from matplotlib.axes import Axes


__all__ = ["ErrorComponent", "ErrorMagnitude", "ErrorStructure"]


############################
# 1/ ERROR MAGNITUDE MODELS
############################


@dataclass(frozen=True)
class ErrorMagnitude:
    """Constant or predictor dependent magnitude of one error component.

    A grouped model interpolates a statistic estimated by :func:`geoutils.stats.grouped_stats`. ``scale`` retains the
    correction obtained by standardizing the complete error proxy. ``variance_offset`` removes variance assigned to
    fixed components while keeping the original grouped observations available for diagnostics.

    :param kind: ``"constant"`` or ``"grouped"`` magnitude model.
    :param value: Constant magnitude when ``kind="constant"``.
    :param predictor_names: Ordered predictor names for a grouped model.
    :param grouped_statistics: Complete grouped statistics table used for interpolation.
    :param value_name: Value level selected from the grouped statistics columns.
    :param statistic: Spread statistic selected from the grouped statistics columns.
    :param min_count: Smallest group count retained for interpolation.
    :param scale: Multiplicative correction applied after interpolation.
    :param variance_offset: Constant variance assigned to other error components.
    :param floor: Smallest predicted magnitude after subtracting ``variance_offset``.
    """

    kind: Literal["constant", "grouped"]
    value: float | None = None
    predictor_names: tuple[str, ...] = ()
    grouped_statistics: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    value_name: str = "error"
    statistic: str = "nmad"
    min_count: int = 0
    scale: float = 1.0
    variance_offset: float = 0.0
    floor: float = 0.0
    _interpolator: Callable[[Mapping[str, Any]], NDArray[np.floating[Any]]] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate magnitude parameters and prepare the interpolator for grouped magnitudes."""

        # Normalize numeric configuration before choosing the magnitude representation
        object.__setattr__(self, "predictor_names", tuple(self.predictor_names))
        for name in ("scale", "variance_offset", "floor"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Magnitude scale must be finite and strictly positive.")

        # Require non-negative removed variance and a finite lower magnitude bound
        if not np.isfinite(self.variance_offset) or self.variance_offset < 0:
            raise ValueError("Magnitude variance_offset must be finite and non-negative.")
        if not np.isfinite(self.floor) or self.floor < 0:
            raise ValueError("Magnitude floor must be finite and non-negative.")

        # Validate constant and grouped models through separate complete paths
        if self.kind == "constant":
            if self.value is None or not np.isfinite(self.value) or self.value < 0:
                raise ValueError("A constant magnitude requires a finite, non-negative value.")
            object.__setattr__(self, "value", float(self.value))
            if self.predictor_names or self.grouped_statistics is not None:
                raise ValueError("A constant magnitude cannot contain grouped predictor metadata.")

        # Require unique predictor names and a grouped table for a variable magnitude model
        elif self.kind == "grouped":
            if self.value is not None:
                raise ValueError("A grouped magnitude cannot also define a constant value.")
            if (
                not self.predictor_names
                or any(not isinstance(name, str) or not name for name in self.predictor_names)
                or len(set(self.predictor_names)) != len(self.predictor_names)
            ):
                raise ValueError("A grouped magnitude requires unique predictor names.")
            if self.grouped_statistics is None:
                raise ValueError("A grouped magnitude requires grouped_statistics.")

            # Copy the small table so external edits cannot invalidate later predictions
            table = self.grouped_statistics.copy(deep=True)
            object.__setattr__(self, "grouped_statistics", table)
            if tuple(table.index.names) != self.predictor_names:
                raise ValueError("Grouped statistic index names must match predictor_names in the same order.")

            # Exclude negative spread estimates from interpolation while preserving the diagnostic table
            column = (self.value_name, self.statistic)
            if column in table:
                table = table.copy()
                table.loc[table[column] < 0, column] = np.nan

            # Build the reusable interpolator after validating the selected predictor axes
            object.__setattr__(
                self,
                "_interpolator",
                interp_binning(
                    table,
                    value_name=self.value_name,
                    statistic=self.statistic,
                    min_count=self.min_count,
                ),
            )
        else:
            raise ValueError("Magnitude kind must be 'constant' or 'grouped'.")

    @classmethod
    def constant(cls, value: float) -> ErrorMagnitude:
        """Create a constant error magnitude.

        :param value: Non-negative error magnitude.
        :returns: Constant magnitude model.
        """

        return cls(kind="constant", value=value)

    @classmethod
    def grouped(
        cls,
        statistics: pd.DataFrame,
        *,
        predictor_names: Sequence[str] | None = None,
        value_name: str = "error",
        statistic: str = "nmad",
        min_count: int = 0,
        scale: float = 1.0,
        variance_offset: float = 0.0,
        floor: float = 0.0,
    ) -> ErrorMagnitude:
        """Create a predictor dependent magnitude from grouped statistics.

        :param statistics: Complete table returned by :func:`geoutils.stats.grouped_stats`.
        :param predictor_names: Ordered predictor names, defaulting to the table index names.
        :param value_name: Value level selected from the grouped statistics columns.
        :param statistic: Spread statistic selected from the grouped statistics columns.
        :param min_count: Smallest group count retained for interpolation.
        :param scale: Multiplicative correction applied after interpolation.
        :param variance_offset: Constant variance assigned to other error components.
        :param floor: Smallest predicted magnitude after subtracting ``variance_offset``.
        :returns: Grouped magnitude model.
        """

        # Infer names from the labelled index while rejecting unnamed predictor dimensions
        raw_names = tuple(statistics.index.names if predictor_names is None else predictor_names)
        if any(not isinstance(name, str) or not name for name in raw_names):
            raise ValueError("Grouped magnitude predictor names must be non-empty strings.")
        names = tuple(name for name in raw_names if isinstance(name, str))
        return cls(
            kind="grouped",
            predictor_names=names,
            grouped_statistics=statistics,
            value_name=value_name,
            statistic=statistic,
            min_count=min_count,
            scale=scale,
            variance_offset=variance_offset,
            floor=floor,
        )

    def predict(self, predictors: Mapping[str, Any] | None = None) -> float | NDArray[np.floating[Any]]:
        """Evaluate the magnitude for optional predictor values.

        :param predictors: Named arrays required by a grouped magnitude model.
        :returns: Constant scalar or array matching the broadcast predictor shape.
        """

        # Return a scalar directly because constant fields need no predictor allocation
        if self.kind == "constant":
            if self.value is None:
                raise AssertionError("A validated constant magnitude must define value.")
            return self.value
        if predictors is None or self._interpolator is None:
            raise ValueError(f"Grouped magnitude requires predictors {self.predictor_names!r}.")

        # Apply the standardization correction before removing fixed component variance
        total = self.scale * self._interpolator(predictors)
        variance = np.maximum(total**2 - self.variance_offset, self.floor**2)
        return np.sqrt(variance)

    @property
    def reference_value(self) -> float:
        """Representative magnitude used for correlation summaries."""

        # Use the exact scalar for a constant component
        if self.kind == "constant":
            if self.value is None:
                raise AssertionError("A validated constant magnitude must define value.")
            return self.value

        # Summarize reliable grouped estimates without evaluating a dense predictor grid
        if self.grouped_statistics is None:
            raise AssertionError("A validated grouped magnitude must contain statistics.")
        values = self.grouped_statistics[(self.value_name, self.statistic)].where(
            self.grouped_statistics[(self.value_name, "count")] >= self.min_count
        )

        # Apply the same variance allocation as local predictions before taking a representative median
        scaled = self.scale * values.to_numpy(dtype=float)
        magnitude = np.sqrt(np.maximum(scaled**2 - self.variance_offset, self.floor**2))
        finite = magnitude[np.isfinite(magnitude)]
        if finite.size == 0:
            raise ValueError("Grouped magnitude contains no finite reference values.")
        return float(np.median(finite))


############################
# 2/ INDEPENDENT COMPONENTS
############################


def _normalize_correlation_model(model: gu.VariogramModel) -> gu.VariogramModel:
    """Remove amplitude from a variogram model while retaining its correlation shape."""

    # Reject nuggets because an independent component represents them without a spatial model
    if model.nugget != 0:
        raise ValueError("Error component correlations cannot contain a nugget; use a separate independent component.")
    if model.sill <= 0:
        raise ValueError("Error component correlations require a strictly positive sill.")

    # Normalize nested structures by their combined sill while retaining relative weights
    if model.components:
        divisor = model.sill if model.model_name == "sum" else model.sill ** (1 / len(model.components))
        components = tuple(
            replace(component, partial_sill=float(component.partial_sill or 0.0) / divisor)
            for component in model.components
        )
        return replace(model, partial_sill=1.0, components=components)
    return replace(model, partial_sill=1.0)


@dataclass(frozen=True)
class ErrorComponent:
    """One independent contribution to an error structure.

    ``correlation=None`` represents spatially independent error. Any supplied variogram is normalized to unit variance
    so that ``magnitude`` remains the only amplitude parameter.

    :param name: Unique descriptive component name.
    :param magnitude: Constant, grouped model or numeric error magnitude.
    :param correlation: Fitted GeoUtils variogram model, variogram result or ``None`` for independent error.
    :param metadata: Additional small diagnostic metadata.
    """

    name: str
    magnitude: ErrorMagnitude | float
    correlation: gu.VariogramModel | gu.Variogram | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Normalize magnitude and correlation shorthand while preserving a component's independent amplitude."""

        # Normalize public shorthand forms before validating component identity
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Error component name must be a non-empty string.")
        if isinstance(self.magnitude, (int, float, np.integer, np.floating)):
            object.__setattr__(self, "magnitude", ErrorMagnitude.constant(float(self.magnitude)))
        elif not isinstance(self.magnitude, ErrorMagnitude):
            raise TypeError("Error component magnitude must be numeric or an ErrorMagnitude.")

        # Extract portable model metadata and normalize its sill to one
        correlation = self.correlation
        if isinstance(correlation, gu.Variogram):
            if correlation.model is None:
                raise ValueError("An error component requires a fitted variogram model.")
            correlation = correlation.model

        # Validate the extracted correlation model before removing its amplitude
        if correlation is not None:
            if not isinstance(correlation, gu.VariogramModel):
                raise TypeError("Error component correlation must be a GeoUtils VariogramModel or Variogram.")
            correlation = _normalize_correlation_model(correlation)

        # Freeze normalized correlation and copied metadata on the immutable component
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def predict_magnitude(self, predictors: Mapping[str, Any] | None = None) -> float | NDArray[np.floating[Any]]:
        """Evaluate this component magnitude.

        :param predictors: Named arrays required by a grouped magnitude model.
        :returns: Constant scalar or array matching the predictor support.
        """

        # Evaluate the validated magnitude model independently of its spatial correlation
        if not isinstance(self.magnitude, ErrorMagnitude):
            raise AssertionError("A validated component must contain an ErrorMagnitude.")
        return self.magnitude.predict(predictors)

    def predict_correlation(self, distance: ArrayLike | float) -> float | NDArray[np.floating[Any]]:
        """Evaluate this component correlation at spatial distances.

        :param distance: Spatial distance or array of distances.
        :returns: Correlation with the same shape as ``distance``.
        """

        # Represent independent errors as one only for coincident observations
        distances = np.asarray(distance, dtype=float)
        if self.correlation is None:
            result = np.where(distances == 0, 1.0, 0.0)
        else:
            result = np.asarray(self.correlation.correlation(distances), dtype=float)

        # Preserve scalar calls while retaining the shape of distance arrays
        return result[()] if distances.ndim == 0 else result


############################
# 3/ COMBINED ERROR STRUCTURE
############################


class ErrorStructure(Mapping[str, ErrorComponent]):
    """Named additive components describing the covariance of an error.

    Components are interpreted as mutually independent. Their variances therefore add, while their simulated random
    fields can be generated separately and summed. The container retains only small fitted results and diagnostics;
    sampled spatial pairs are discarded during estimation.

    :param components: Named mapping or sequence of error components.
    :param empirical_variogram: Standardized empirical variogram used during fitting.
    :param fit_diagnostics: Small optimizer and identifiability diagnostics.
    :param metadata: Additional context such as estimator settings and source assumptions.
    """

    def __init__(
        self,
        components: Mapping[str, ErrorComponent] | Sequence[ErrorComponent],
        *,
        empirical_variogram: gu.Variogram | None = None,
        fit_diagnostics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Construct a named collection of independent error components and their estimation diagnostics."""

        # Normalize component sequences to an insertion ordered mapping
        if isinstance(components, Mapping):
            normalized = dict(components)
            if not normalized or any(not isinstance(component, ErrorComponent) for component in normalized.values()):
                raise ValueError("ErrorStructure requires at least one ErrorComponent.")
            if any(name != component.name for name, component in normalized.items()):
                raise ValueError("Error component mapping keys must match component names.")

        # Accept an ordered component sequence and reject names that would overwrite another component
        else:
            if not components or any(not isinstance(component, ErrorComponent) for component in components):
                raise ValueError("ErrorStructure requires at least one ErrorComponent.")
            normalized = {component.name: component for component in components}
            if len(normalized) != len(components):
                raise ValueError("Error component names must be unique.")

        # Freeze the container mappings while retaining lightweight fit objects directly
        self._components = MappingProxyType(normalized)
        self.empirical_variogram = empirical_variogram
        self.fit_diagnostics = MappingProxyType(dict(fit_diagnostics or {}))
        self.metadata = MappingProxyType(dict(metadata or {}))

    def __getitem__(self, name: str) -> ErrorComponent:
        """Return the component with the requested name."""

        return self._components[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over component names in their declared order."""

        return iter(self._components)

    def __len__(self) -> int:
        """Return the number of independent error components."""

        return len(self._components)

    def __repr__(self) -> str:
        """Return the component summary used for interactive inspection."""

        return cast(str, self.info(verbose=False))

    @property
    def components(self) -> Mapping[str, ErrorComponent]:
        """Read only mapping of components by name."""

        return self._components

    @classmethod
    def estimate(
        cls,
        error_proxy: Any,
        *,
        predictors: Mapping[str, Any] | None = None,
        components: Mapping[str, Mapping[str, Any]] | None = None,
        mask: Any | None = None,
        bins: Mapping[str, Any] | int | None = None,
        spread_estimator: Callable[[Any], Any] = gu.stats.nmad,
        min_count: int = 100,
        outlier_factor: float | None = 7,
        subsample_magnitude: int | float = 1_000_000,
        variogram_estimator: str | Callable[[Any], float] = "dowd",
        n_pairs: int = 1_000_000,
        pair_sampling: Literal["loglag", "random_xy"] = "loglag",
        n_lags: int = 24,
        min_lag: float | None = None,
        max_lag: float | None = None,
        n_runs: int = 1,
        fit_method: Literal["variogram"] = "variogram",
        refine: bool = True,
        fit_kwargs: Mapping[str, Any] | None = None,
        pair_sampling_kwargs: Mapping[str, Any] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> ErrorStructure:
        """Estimate component magnitudes and correlations from one error proxy.

        At most one component can have ``magnitude="heteroscedastic"`` in the first variogram estimator. Remaining
        components have constant magnitudes. Their initial variance fractions come from a nested standardized
        variogram and are refined against conditional pair semivariances.

        :param error_proxy: Raster or point cloud whose values represent errors.
        :param predictors: Named continuous variables controlling a heteroscedastic magnitude.
        :param components: Ordered named component specifications with ``magnitude`` and ``correlation`` entries.
        :param mask: Spatial or Boolean mask identifying values used for estimation.
        :param bins: Group definitions by predictor, or one bin count applied to every predictor.
        :param spread_estimator: Robust or classical estimator of error magnitude.
        :param min_count: Smallest grouped sample retained in the magnitude model.
        :param outlier_factor: Standardized error threshold used for a second magnitude scaling, or ``None``.
        :param subsample_magnitude: Maximum observations used for grouped magnitude statistics.
        :param variogram_estimator: Empirical variogram estimator name or callable.
        :param n_pairs: Target spatial pairs per variogram run and conditional refinement.
        :param pair_sampling: Pair sampling scheme exposed by GeoUtils spatial objects.
        :param n_lags: Number of empirical lag classes.
        :param min_lag: Smallest sampled spatial distance.
        :param max_lag: Largest sampled spatial distance.
        :param n_runs: Independent variogram samples used to estimate empirical sampling error.
        :param fit_method: Fitting backend, currently ``"variogram"``.
        :param refine: Whether to refine component contributions using conditional pair semivariances.
        :param fit_kwargs: Options passed to :meth:`geoutils.Variogram.fit`.
        :param pair_sampling_kwargs: Advanced options passed to ``sample_pairs`` and ``variogram``.
        :param random_state: Random generator or seed used throughout estimation.
        :returns: Fitted error structure with compact diagnostics.
        """

        # Import fitting lazily so ordinary construction avoids SciPy optimizer setup
        from xdem.uncertainty.estimation import _estimate_error_structure

        return _estimate_error_structure(
            error_proxy,
            predictors=predictors,
            components=components,
            mask=mask,
            bins=bins,
            spread_estimator=spread_estimator,
            min_count=min_count,
            outlier_factor=outlier_factor,
            subsample_magnitude=subsample_magnitude,
            variogram_estimator=variogram_estimator,
            n_pairs=n_pairs,
            pair_sampling=pair_sampling,
            n_lags=n_lags,
            min_lag=min_lag,
            max_lag=max_lag,
            n_runs=n_runs,
            fit_method=fit_method,
            refine=refine,
            fit_kwargs=fit_kwargs,
            pair_sampling_kwargs=pair_sampling_kwargs,
            random_state=random_state,
        )

    def refit(
        self,
        correlation_models: str | Sequence[str] | None = None,
        *,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> ErrorStructure:
        """Refit correlations and their component magnitude contributions.

        Refit uses the retained empirical variogram and grouped magnitude model. It updates every dependent component
        together, avoiding an inconsistent magnitude fit paired with a new correlation fit. Conditional pair
        refinement requires re-estimation because individual pairs are deliberately not stored.

        :param correlation_models: Ordered variogram models, defaulting to the current correlated components.
        :param fit_kwargs: Options passed to :meth:`geoutils.Variogram.fit`.
        :returns: New error structure fitted from the retained compact diagnostics.
        """

        # Keep fitting in the estimation module and rebuild all dependent component contributions together
        from xdem.uncertainty.estimation import _refit_error_structure

        return _refit_error_structure(
            self,
            correlation_models=None if correlation_models is None else correlation_models,
            fit_kwargs=fit_kwargs,
        )

    def predict_magnitude(
        self,
        predictors: Mapping[str, Any] | None = None,
        *,
        component: str | None = None,
        like: Any | None = None,
    ) -> Any:
        """Evaluate one component magnitude or their combined error magnitude.

        :param predictors: Named values required by grouped component magnitudes.
        :param component: Optional component name; all component variances are combined when omitted.
        :param like: Optional raster or point cloud whose type and support wrap the result.
        :returns: Scalar, array or spatial object containing error magnitudes.
        """

        # Evaluate one requested component or combine independent component variances
        if component is not None:
            result: Any = self[component].predict_magnitude(predictors)
        else:
            variances: Any = 0.0
            for error_component in self.values():
                magnitude = error_component.predict_magnitude(predictors)
                variances = variances + np.asarray(magnitude) ** 2
            result = np.sqrt(variances)

        # Preserve scalar results unless a target spatial support was requested
        if like is None:
            return float(result) if np.ndim(result) == 0 else result
        if np.ndim(result) == 0:
            support_shape = like.shape if hasattr(like, "ij2xy") else (like.point_count,)
            result = np.full(support_shape, float(result), dtype=float)

        # Preserve the target raster mask even when a constant magnitude fills its complete shape
        result_array = np.asarray(result)
        if hasattr(like, "ij2xy"):
            result_array = np.ma.masked_array(
                result_array,
                mask=np.asarray(like.get_mask()).squeeze() | ~np.isfinite(result_array),
            )
        return like.copy(new_array=result_array)

    def predict_variance(
        self,
        predictors: Mapping[str, Any] | None = None,
        *,
        component: str | None = None,
    ) -> float | NDArray[np.floating[Any]]:
        """Evaluate one component variance or their combined variance.

        :param predictors: Named values required by grouped component magnitudes.
        :param component: Optional component name; all component variances are combined when omitted.
        :returns: Scalar or array of error variance.
        """

        # Square the combined magnitude so independent contributions retain their variance sum
        magnitude = self.predict_magnitude(predictors, component=component)
        variance = np.asarray(magnitude, dtype=float) ** 2
        return float(variance) if variance.ndim == 0 else cast(NDArray[np.floating[Any]], variance)

    def predict_correlation(self, distance: ArrayLike | float) -> float | NDArray[np.floating[Any]]:
        """Evaluate the representative variance weighted correlation.

        :param distance: Spatial distance or array of distances.
        :returns: Correlation summarized across fitted component magnitudes.
        """

        # Weight unit variance correlations by representative component variances
        total_variance = sum(
            cast(ErrorMagnitude, component.magnitude).reference_value ** 2 for component in self.values()
        )
        if total_variance <= 0:
            raise ValueError("An error structure with zero total magnitude has no defined correlation.")

        # Combine normalized correlations using each component's representative variance
        weighted = np.zeros_like(np.asarray(distance, dtype=float))
        for component in self.values():
            magnitude = cast(ErrorMagnitude, component.magnitude)
            weighted += magnitude.reference_value**2 * component.predict_correlation(distance)
        result = weighted / total_variance
        return float(result) if result.ndim == 0 else result

    def predict_covariance(
        self,
        distance: ArrayLike | float,
        *,
        predictors: Mapping[str, Any] | None = None,
        other_predictors: Mapping[str, Any] | None = None,
    ) -> float | NDArray[np.floating[Any]]:
        """Evaluate covariance for corresponding pairs of locations.

        :param distance: Distance for each location pair.
        :param predictors: Predictor values at the first endpoint of each pair.
        :param other_predictors: Predictor values at second endpoints, defaulting to the first values.
        :returns: Covariance for each location pair.
        """

        # Multiply endpoint magnitudes before adding independent component covariances
        if other_predictors is None:
            other_predictors = predictors

        # Evaluate both local magnitudes because a variable component can differ between the two locations
        covariance: Any = 0.0
        for component in self.values():
            first = component.predict_magnitude(predictors)
            second = component.predict_magnitude(other_predictors)
            covariance = covariance + np.asarray(first) * np.asarray(second) * component.predict_correlation(distance)

        # Preserve scalar or array output according to the covariance calculation
        result = np.asarray(covariance)
        return result[()] if result.ndim == 0 else result

    def to_covariance_matrix(
        self,
        coordinates: ArrayLike,
        *,
        predictors: Mapping[str, Any] | None = None,
        max_points: int = 2_000,
    ) -> NDArray[np.floating[Any]]:
        """Build a dense covariance matrix for explicit coordinates.

        :param coordinates: Coordinate array shaped as observations by dimensions.
        :param predictors: Named predictor arrays with one value per observation.
        :param max_points: Safety limit guarding accidental allocation of very large dense matrices.
        :returns: Dense covariance matrix including all component contributions.
        """

        # Validate the quadratic allocation before computing pair distances
        coordinate_array = np.asarray(coordinates, dtype=float)
        if coordinate_array.ndim != 2 or len(coordinate_array) == 0 or np.any(~np.isfinite(coordinate_array)):
            raise ValueError("coordinates must be a finite two dimensional observation array.")
        if len(coordinate_array) > max_points:
            raise ValueError(
                f"Dense covariance for {len(coordinate_array)} points exceeds max_points={max_points}; "
                "use a lazy covariance backend instead."
            )

        # Compute distances once because every component uses the same observation pairs
        distances = cdist(coordinate_array, coordinate_array)

        # Add component covariance matrices with outer products of local magnitudes
        covariance = np.zeros_like(distances)
        for component in self.values():
            magnitude = np.asarray(component.predict_magnitude(predictors), dtype=float)
            if magnitude.ndim == 0:
                magnitude = np.full(len(coordinate_array), float(magnitude))
            magnitude = magnitude.squeeze()
            if magnitude.ndim != 1 or len(magnitude) != len(coordinate_array):
                raise ValueError("Each magnitude predictor must yield one value per coordinate.")

            # Multiply each pair of local magnitudes by their component correlation before adding contributions
            covariance += np.outer(magnitude, magnitude) * component.predict_correlation(distances)
        return covariance

    @overload
    def generate_random_field(
        self,
        like: Any,
        *,
        predictors: Mapping[str, Any] | None = None,
        n_fields: Literal[1] = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> Any: ...

    @overload
    def generate_random_field(
        self,
        like: Any,
        *,
        predictors: Mapping[str, Any] | None = None,
        n_fields: int,
        random_state: int | np.random.Generator | None = None,
    ) -> list[Any]: ...

    def generate_random_field(
        self,
        like: Any,
        *,
        predictors: Mapping[str, Any] | None = None,
        n_fields: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> Any | list[Any]:
        """Generate independent error realizations on raster or point cloud support.

        :param like: Raster or point cloud defining output coordinates and type.
        :param predictors: Named values required by grouped component magnitudes on ``like``.
        :param n_fields: Number of independent realizations.
        :param random_state: Random generator or seed used for reproducible fields.
        :returns: One spatial object or a list when ``n_fields`` is greater than one.
        """

        # Load the numerical workflow only when random field generation is requested
        from xdem.uncertainty.numerical import generate_random_field

        return generate_random_field(like, self, predictors=predictors, n_fields=n_fields, random_state=random_state)

    def plot_correlation(self, ax: Axes | None = None, *, show_error: bool = True, **kwargs: Any) -> Axes:
        """Plot the empirical and combined fitted variogram.

        :param ax: Existing Matplotlib axes, or ``None`` to create one.
        :param show_error: Whether to draw finite empirical sampling errors.
        :param kwargs: Keyword arguments passed to the empirical point plot.
        :returns: Axes containing the variogram diagnostics.
        """

        # Require fitted observations before delegating their display to GeoUtils
        if self.empirical_variogram is None:
            raise ValueError("No empirical variogram is stored on this error structure.")
        return self.empirical_variogram.plot(ax=ax, show_error=show_error, **kwargs)

    def plot_magnitude(
        self,
        *,
        component: str | None = None,
        min_count: int | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Plot grouped magnitude statistics and sample counts.

        :param component: Grouped component to plot, inferred when there is only one.
        :param min_count: Smallest count shown, defaulting to the fitted threshold.
        :param kwargs: Keyword arguments passed to :func:`geoutils.stats.plot_grouped_stats`.
        :returns: Named plotting axes.
        """

        # Select the only grouped component unless the caller names one explicitly
        grouped = [
            value
            for value in self.values()
            if isinstance(value.magnitude, ErrorMagnitude) and value.magnitude.kind == "grouped"
        ]
        selected: ErrorComponent | None
        if component is not None:
            selected = self[component]
        elif len(grouped) == 1:
            selected = grouped[0]
        else:
            selected = None
        if (
            selected is None
            or not isinstance(selected.magnitude, ErrorMagnitude)
            or selected.magnitude.kind != "grouped"
        ):
            raise ValueError("Select one grouped error component to plot.")

        # Require the retained bin observations needed to display this component's magnitude
        magnitude = selected.magnitude
        if magnitude.grouped_statistics is None:
            raise AssertionError("A validated grouped magnitude must contain statistics.")
        # Plot the selected component after applying its scale and allocated fixed variance
        table = magnitude.grouped_statistics.copy(deep=True)
        statistic_column = (magnitude.value_name, magnitude.statistic)
        values = magnitude.scale * table[statistic_column].to_numpy(dtype=float)
        table[statistic_column] = np.sqrt(np.maximum(values**2 - magnitude.variance_offset, magnitude.floor**2))

        # Use the fitted count threshold unless the caller requests a different display threshold
        return gu.stats.plot_grouped_stats(
            table,
            value=magnitude.value_name,
            statistic=magnitude.statistic,
            min_count=magnitude.min_count if min_count is None else min_count,
            **kwargs,
        )

    def plot(self, **kwargs: Any) -> Mapping[str, Any]:
        """Plot every available magnitude and correlation diagnostic.

        :param kwargs: Keyword arguments passed to the correlation plot.
        :returns: Mapping containing the created plotting axes.
        """

        # Keep independent figures because grouped statistics manage their own panel layout
        axes: dict[str, Any] = {}
        grouped = [
            component
            for component in self.values()
            if isinstance(component.magnitude, ErrorMagnitude) and component.magnitude.kind == "grouped"
        ]

        # Create one magnitude figure for each grouped component and an optional correlation figure
        for component in grouped:
            axes[f"magnitude:{component.name}"] = self.plot_magnitude(component=component.name)
        if self.empirical_variogram is not None:
            axes["correlation"] = self.plot_correlation(**kwargs)

        # Report when a manually constructed structure contains no fitted observations to display
        if not axes:
            warnings.warn("This error structure contains no empirical diagnostics to plot.", UserWarning)
        return axes

    def info(self, *, verbose: bool = True) -> str | None:
        """Summarize component magnitudes and correlation models.

        :param verbose: Whether to print the summary instead of returning it.
        :returns: Summary string when ``verbose=False``.
        """

        # Build one compact line per component for interactive inspection
        lines = [f"ErrorStructure with {len(self)} independent component(s)"]
        for component in self.values():
            model = "independent" if component.correlation is None else component.correlation.model_name
            error_range = None if component.correlation is None else component.correlation.effective_range
            magnitude = cast(ErrorMagnitude, component.magnitude)

            # Describe constant and predictor-dependent magnitudes using the corresponding fitted information
            magnitude_text = (
                f"constant {magnitude.reference_value:.4g}"
                if magnitude.kind == "constant"
                else f"grouped by {', '.join(magnitude.predictor_names)}"
            )
            range_text = "" if error_range is None else f", range {error_range:.4g}"
            lines.append(f"  {component.name}: {magnitude_text}, {model}{range_text}")

        # Return the same summary text used by representation, or print it for interactive use
        text = "\n".join(lines)
        if verbose:
            print(text)
            return None
        return text

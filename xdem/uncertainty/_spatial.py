# Copyright (c) 2026 xDEM developers
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

from __future__ import annotations

import warnings
from typing import Any, Iterable, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

import geoutils as gu
from geoutils.raster.georeferencing import _coords, _res
from geoutils.raster.array import get_array_and_mask

from xdem._misc import import_optional
from xdem.uncertainty._metricspace import RegularLogLagMetricSpace, IrregularLogLagMetricSpace

import gstools as gs


def _stable_rescale(alpha: float) -> float:
    # skgstat stable -> gstools Stable: rescale = 3^(1/shape)
    return float(np.power(3.0, 1.0 / alpha))


# Mirrors skgstat/interfaces/gstools.py MODEL_MAP coefficients:
# exponential: rescale=3, gaussian: rescale=2, matern: rescale=4, stable: rescale=3^(1/alpha)
# spherical/cubic: rescale=1 (default) :contentReference[oaicite:1]{index=1}
_SKGSTAT_TO_GSTOOLS = {
    "spherical": dict(cls=gs.Spherical, rescale=1.0),
    "exponential": dict(cls=gs.Exponential, rescale=3.0),
    "gaussian": dict(cls=gs.Gaussian, rescale=2.0),
    "cubic": dict(cls=gs.Cubic, rescale=1.0),
    "matern": dict(cls=gs.Matern, rescale=4.0),   # plus nu = smoothness :contentReference[oaicite:2]{index=2}
    "stable": dict(cls=gs.Stable, rescale=_stable_rescale),  # plus alpha = shape :contentReference[oaicite:3]{index=3}
}


def params_to_gstools_model(params: pd.DataFrame, *, dim: int = 2) -> gs.CovModel:
    """
    Convert a sum of SciKit-GStat-like variogram parameter rows into a GSTools CovModel sum,
    applying the same rescale coefficients as skgstat_to_gstools.

    Expected columns per row:
      - "model": model name (spherical/exponential/gaussian/cubic/stable/matern)
      - "range": effective range (will be passed as gstools len_scale, with rescale applied)
      - "psill": partial sill (will be passed as gstools var)
      - "smooth": for matern (nu) and stable (alpha/shape)
    """
    model_sum: gs.CovModel | None = None

    for _, row in params.iterrows():
        name = str(row["model"]).lower()
        spec = _SKGSTAT_TO_GSTOOLS[name]

        len_scale = float(row["range"])
        var = float(row["psill"])

        kwargs = dict(dim=dim, var=var, len_scale=len_scale)

        # Apply rescale factor (constant or function of smooth parameter)
        rescale = spec["rescale"]
        if callable(rescale):
            # Stable: alpha/shape stored in "smooth" in your table
            alpha = float(row["smooth"])
            kwargs["rescale"] = rescale(alpha)
            kwargs["alpha"] = alpha
        else:
            kwargs["rescale"] = float(rescale)

        # Matern: smoothness stored in "smooth" -> gstools "nu"
        if name == "matern":
            kwargs["nu"] = float(row["smooth"])

        m = spec["cls"](**kwargs)
        model_sum = m if model_sum is None else (model_sum + m)

    return model_sum

def _default_log_bin_edges(*, min_lag: float, max_lag: float, n_bins: int = 24) -> np.ndarray:
    """
    Default log-lag binning (right edges) for variography.

    We use geometric spacing (logspace) to ensure adequate sampling at short and long lags.
    """
    if not np.isfinite(min_lag) or not np.isfinite(max_lag) or min_lag <= 0 or max_lag <= 0:
        raise ValueError("min_lag and max_lag must be finite and > 0.")
    if max_lag <= min_lag:
        raise ValueError("max_lag must be > min_lag.")
    if n_bins is None or n_bins < 2:
        raise ValueError("n_bins must be >= 2 for default log-lag binning.")

    edges = np.logspace(np.log10(min_lag), np.log10(max_lag), int(n_bins))
    edges[-1] = float(max_lag)
    return edges.astype(float)


def _filter_kwargs_for_call(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter keyword arguments to those accepted by a callable (constructor or function).

    This keeps the public API flexible (users can pass through arguments for both MetricSpace and Variogram),
    while warning/error behavior remains under control.
    """
    try:
        import inspect

        sig = inspect.signature(func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        # If introspection fails, pass everything and let it error upstream.
        return dict(kwargs)


def _coords_from_transform(
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    aop: Literal["Area", "Point"] | None,
) -> np.ndarray:
    """
    Build (N, 2) coordinates for a regular grid using GeoUtils `_coords`.
    """
    xx, yy = _coords(transform=transform, shape=shape, area_or_point=aop)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _as_irregular_inputs_from_gdf(gdf: gpd.GeoDataFrame, *, z_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (coords, values) from a GeoDataFrame of points.

    :param gdf: Point GeoDataFrame.
    :param z_name: Column name holding the values to variogram.
    :returns: coords (N,2), values (N,)
    """
    if z_name not in gdf.columns:
        raise ValueError(f"GeoDataFrame is missing column '{z_name}'.")
    if gdf.geometry is None:
        raise ValueError("GeoDataFrame has no geometry column.")
    if not gdf.geometry.geom_type.isin(["Point"]).all():
        raise TypeError("GeoDataFrame geometry must contain only Point geometries.")

    vals = np.asarray(gdf[z_name].to_numpy())
    coords = np.column_stack((gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()))
    return coords, vals


def _compute_maxlag_from_extent(extent: tuple[float, float, float, float]) -> float:
    """Compute diagonal max lag from (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = extent
    return float(np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2))


def _variogram(
    values: gu.Raster | gpd.GeoDataFrame | np.ndarray,
    *,
    # Optional spatial metadata (array inputs)
    transform: rio.transform.Affine | None = None,
    coords: tuple[np.ndarray, np.ndarray] | None = None,
    crs: rio.crs.CRS | None = None,
    area_or_point: Literal["Area", "Point"] | None = None,
    z_name: str = "z",
    # Sampling backend
    sampling: Literal["loglag", "random_xy"] = "loglag",
    # Pair sample target (interpreted by the MetricSpace)
    samples: int = 1_000_000,
    # Distance range (required for irregular log-lag; optional otherwise)
    min_dist: float | None = None,
    max_dist: float | None = None,
    # Default binning (log-lag)
    bin_func: Iterable[float] | None = None,
    n_bins: int = 24,
    # Variogram estimator (scikit-gstat)
    estimator: Literal["matheron", "cressie", "genton", "dowd"] = "dowd",
    # Random seed/state
    random_state: int | None = None,
    # Pass-through kwargs to MetricSpace + Variogram
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Sample an empirical variogram using MetricSpace-based pair sampling.

    This helper supports:
      - Regular grids (Raster or 2D array + transform):
          * sampling="loglag"   -> RegularLogLagMetricSpace
          * sampling="random_xy"-> scikit-gstat ProbabilisticMetricSpace
      - Irregular points (GeoDataFrame or 1D array + coords):
          * sampling="loglag"   -> IrregularLogLagMetricSpace
          * sampling="random_xy"-> scikit-gstat ProbabilisticMetricSpace

    The default binning is logarithmic in lag (right bin edges), suitable for variography across short and long ranges.

    :param values: Raster/DEM, GeoDataFrame of points, or array.
    :param transform: Affine transform for 2D array inputs.
    :param coords: Coordinates (N,2) for 1D array inputs.
    :param crs: CRS for consistency checks (only enforced for GeoDataFrame inputs if provided).
    :param area_or_point: Raster pixel interpretation when deriving coordinates (Area/Point).
    :param z_name: Value column name for GeoDataFrame inputs.
    :param sampling: Sampling backend, "loglag" (default) or "random_xy".
    :param samples: Target number of sampled pairs (exact behavior depends on MetricSpace implementation).
    :param min_dist: Minimum lag distance (required for irregular log-lag).
    :param max_dist: Maximum lag distance (defaults to extent diagonal when possible).
    :param bin_func: Bin edges (right edges). Defaults to log-spaced edges in [min_dist, max_dist].
    :param n_bins: Number of default log bins (ignored if bin_func is provided).
    :param estimator: Empirical variogram estimator used by scikit-gstat.
    :param random_state: Random seed for sampling.
    :param kwargs: Passed through to MetricSpace and Variogram (filtered by signature).
    :return: DataFrame with columns (lags, exp, err_exp, count).
    """
    skg = import_optional("skgstat", package_name="scikit-gstat")

    if sampling not in ("loglag", "random_xy"):
        raise ValueError("sampling must be either 'loglag' or 'random_xy'.")

    # ---------------------------------------------------------------------
    # 1) Normalize inputs and decide regular vs irregular sampling mode
    # ---------------------------------------------------------------------

    is_raster = isinstance(values, gu.Raster)
    is_gdf = isinstance(values, gpd.GeoDataFrame)

    if is_raster:
        arr, _ = get_array_and_mask(values)
        arr = np.asarray(arr).squeeze()
        if arr.ndim != 2:
            raise ValueError("Raster values must be 2D after squeezing.")
        if transform is None:
            transform = values.transform
        if crs is None:
            crs = values.crs
        if area_or_point is None:
            area_or_point = getattr(values, "area_or_point", None)

        dx, dy = _res(transform)
        dx = float(abs(dx))
        dy = float(abs(dy))
        is_regular = True

    elif is_gdf:
        gdf = values
        if crs is not None and gdf.crs is not None and gdf.crs != crs:
            raise ValueError("GeoDataFrame CRS differs from 'crs'; reproject before calling _variogram().")

        coords_i, vals_i = _as_irregular_inputs_from_gdf(gdf, z_name=z_name)
        is_regular = False

    else:
        arr = np.asarray(values).squeeze()
        if arr.ndim == 2:
            if transform is None:
                raise ValueError("For 2D ndarray input, 'transform' must be provided.")
            dx, dy = _res(transform)
            dx = float(abs(dx))
            dy = float(abs(dy))
            is_regular = True
        elif arr.ndim == 1:
            if coords is None:
                raise ValueError(
                    "For 1D ndarray input, 'coords' must be provided as a tuple of (x, y) arrays."
                )

            if not isinstance(coords, tuple) or len(coords) != 2:
                raise ValueError("'coords' must be a tuple of two arrays: (x, y).")

            x, y = coords
            x = np.asarray(x)
            y = np.asarray(y)

            if x.ndim != 1 or y.ndim != 1:
                raise ValueError("Each element of 'coords' must be a 1D array.")

            if x.shape[0] != y.shape[0]:
                raise ValueError("Coordinate arrays 'x' and 'y' must have the same length.")

            if x.shape[0] != arr.shape[0]:
                raise ValueError(
                    "Length of coordinate arrays must match length of 1D values array."
                )

            # Build internal (N, 2) coordinate array for downstream use
            coords_i = np.column_stack((x, y))

            vals_i = arr
            is_regular = False
        else:
            raise ValueError("ndarray input must be 1D or 2D.")

    # ---------------------------------------------------------------------
    # 2) Infer distance range and build default log-lag bins if needed
    # ---------------------------------------------------------------------

    # Regular grid
    if is_regular:
        # Default minimum lag: one-pixel diagonal (consistent with grid sampling)
        if min_dist is None:
            min_dist = float(np.sqrt(dx * dx + dy * dy) / 2)

        # Default maximum lag: extent diagonal derived from transform + shape
        if max_dist is None:
            ny, nx = arr.shape  # rows, cols
            coords_grid = _coords_from_transform(transform, (ny, nx), area_or_point)
            extent = (
                float(coords_grid[:, 0].min()),
                float(coords_grid[:, 0].max()),
                float(coords_grid[:, 1].min()),
                float(coords_grid[:, 1].max()),
            )
            max_dist = _compute_maxlag_from_extent(extent)

    # Irregular grid
    else:

        extent = (
            float(coords_i[:, 0].min()),
            float(coords_i[:, 0].max()),
            float(coords_i[:, 1].min()),
            float(coords_i[:, 1].max()),
        )
        # Default minimum lag: based on average pixel density in the extent
        if sampling == "loglag" and min_dist is None:
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            A = max(dx * dy, 0.0)
            N = len(coords_i)
            s = np.sqrt(A / max(N, 1))
            min_dist = max(1e-12, 0.5 * s)

        # Default max lag: based on extent
        if max_dist is None:
            max_dist = _compute_maxlag_from_extent(extent)

    if min_dist is None or max_dist is None:
        raise AssertionError("min_dist/max_dist must be defined at this stage.")

    if bin_func is None:
        # Provide stable, explicit bin edges to avoid run-to-run bin drift.
        bin_func = _default_log_bin_edges(min_lag=float(min_dist), max_lag=float(max_dist), n_bins=n_bins)

    # ---------------------------------------------------------------------
    # 3) Prepare data (finite mask)
    # ---------------------------------------------------------------------

    if is_regular:
        # For regular grids, keep array form: the MetricSpace handles indexing / NaNs internally.
        vals_for_var = arr
    else:
        finite = np.isfinite(vals_i) & np.isfinite(coords_i).all(axis=1)
        coords_i = coords_i[finite]
        vals_i = vals_i[finite]
        if vals_i.size == 0:
            raise ValueError("No finite values in input.")
        vals_for_var = vals_i

    # ---------------------------------------------------------------------
    # 4) Instantiate MetricSpace and Variogram (kwargs filtered by signature)
    # ---------------------------------------------------------------------

    var_kwargs = dict(
        estimator=estimator,
        bin_func=bin_func,
        maxlag=float(max_dist),
        normalize=False,
        fit_method=None,
    )
    user_kwargs = dict(kwargs)

    # MetricSpace choice:
    # - loglag: dispatch regular/irregular to your custom samplers
    # - random_xy: use scikit-gstat ProbabilisticMetricSpace (regular or irregular)
    if sampling == "loglag":
        if is_regular:
            ms_ctor = RegularLogLagMetricSpace
            ms_kwargs = dict(
                array=arr,
                dx=float(dx),
                dy=float(dy),
                samples=int(samples),
                min_dist=float(min_dist),
                max_dist=float(max_dist),
                seed=random_state,  # your class uses "seed"
            )
        else:
            ms_ctor = IrregularLogLagMetricSpace
            ms_kwargs = dict(
                coords=coords_i,
                samples=int(samples),
                min_dist=float(min_dist),
                max_dist=float(max_dist),
                seed=random_state,  # your class uses "seed"
            )

    else:
        # scikit-gstat spelling differs across versions: keep compatibility by probing attributes.
        ms_ctor = getattr(skg, "ProbabilisticMetricSpace", None) or getattr(skg, "ProbabalisticMetricSpace", None)
        if ms_ctor is None:
            raise ImportError(
                "Could not find scikit-gstat ProbabilisticMetricSpace. "
                "Please update scikit-gstat or adjust the class name mapping."
            )
        ms_kwargs = dict(
            coords=(coords_i if not is_regular else _coords_from_transform(transform, arr.shape, area_or_point)),
            samples=int(samples),
            rnd=random_state,  # scikit-gstat MetricSpace typically uses 'rnd'
        )

    # Filter user kwargs for MetricSpace and Variogram separately
    ms_kwargs.update(_filter_kwargs_for_call(ms_ctor.__init__, user_kwargs))
    var_kwargs.update(_filter_kwargs_for_call(skg.Variogram.__init__, user_kwargs))

    # Instantiate
    M = ms_ctor(**ms_kwargs)
    V = skg.Variogram(M, values=vals_for_var.ravel(), **var_kwargs)

    # ---------------------------------------------------------------------
    # 5) Export empirical variogram (lags, exp, count)
    # ---------------------------------------------------------------------
    bins, exp = V.get_empirical(bin_center=False)
    count = V.bin_count

    df = pd.DataFrame({"lags": bins, "exp": exp, "count": count})
    df["err_exp"] = np.nan

    # ---------------------------------------------------------------------
    # 6) Dowd correction for old scikit-gstat (kept from previous implementation)
    # ---------------------------------------------------------------------

    try:
        from packaging.version import Version

        if Version(skg.__version__) <= Version("1.0.0") and estimator == "dowd":
            df["exp"] = df["exp"] / 2.0
            df["err_exp"] = df["err_exp"] / 2.0
    except Exception:
        pass

    # Remove last bin (often undersampled at the very largest lag)
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    df = df.astype({"exp": "float64", "err_exp": "float64", "lags": "float64", "count": "int64"})

    return df
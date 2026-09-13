(uncertainty-migration)=
# Migrating uncertainty workflows

The entire `xdem.spatialstats` module is deprecated. Its public functions retain their original implementations,
arguments and return formats and emit a `DeprecationWarning` when called. Importing xDEM does not emit a warning.
`DEM.estimate_uncertainty()` is also deprecated. No module-wide removal version is assigned yet;
`spatialstats.nmad()` retains its previously announced removal in version 0.4.

New workflows use an `ErrorStructure`, which stores named independent components. Each component combines a magnitude
model with a correlation model normalized to unit variance. Magnitudes combine as variances, and propagation uses
each component's covariance separately.

## Where functions now live

| Former function or responsibility | Replacement |
| --- | --- |
| `spatialstats.nmad` | `geoutils.stats.nmad` |
| `spatialstats.nd_binning` | `geoutils.stats.grouped_stats` or a spatial object's `.grouped_stats()` |
| `spatialstats.interp_nd_binning` | `xdem.fit.interp_binning` |
| `spatialstats.get_perbin_nd_binning` | `xdem.fit.get_perbin_binning` |
| `spatialstats.two_step_standardization` | `xdem.uncertainty.estimation.two_step_standardization` |
| Magnitude and correlation inference | `ErrorStructure.estimate` or `estimate_error_structure` |
| `spatialstats.sample_empirical_variogram` | Raster or point cloud `.variogram()` |
| `spatialstats.fit_sum_model_variogram` | `geoutils.Variogram.fit` |
| Variogram, covariance and correlation functions | `geoutils.VariogramModel` methods |
| Effective sample size and area uncertainty | `xdem.uncertainty.number_effective_samples`, `spatial_error_propagation` |
| Exact and approximate covariance sums, circular formulas | `xdem.uncertainty.analytical` |
| Random fields, Monte Carlo and patches | `xdem.uncertainty.numerical` |
| `spatialstats.patches_method` | `xdem.uncertainty.patches_method` |
| `spatialstats.convolution`, `mean_filter_nan` | `geoutils.filters.convolution`, `mean_filter_nan` |
| Binning and variogram plots | `geoutils.stats.plot_grouped_stats`, `geoutils.Variogram.plot` |

`xdem.cosampling` and the private uncertainty variography and metric-space implementations existed only on the
development branch. They have been removed. Use GeoUtils `.cosample()` and `.sample_pairs()` directly. A cosampling
result is a raster or point cloud on the support selected by `at`; accessor calls return Xarray or GeoPandas.
Raster bands or point columns contain `"self"`, `"other"`, then auxiliaries in mapping order. Use `.split_bands()`,
`.data` or `.ds` for the usual spatial and array workflows.

## Estimate an error structure

Previously:

```python
sigma, correlation = dem.estimate_uncertainty(other_dem, stable_terrain=stable_mask)
```

Now:

```python
structure = dem.estimate_error_structure(
    other_dem, stable_terrain=stable_mask, other_error="negligible", random_state=42,
)
predictors = {"slope": dem.slope(), "max_curvature": dem.max_curvature()}
sigma = structure.predict_magnitude(predictors, like=dem)
correlation = structure.predict_correlation
```

The `other_error="same"` assumption divides differences by the square root of two; it assumes equal independent
errors in the two datasets. `"negligible"` assigns the difference errors to the source dataset. The former
`precision_of_other="finer"` maps to `other_error="negligible"`.

For an existing error proxy, use `xdem.ErrorStructure.estimate(proxy, predictors=..., mask=...)`. To estimate only
variable magnitudes, use a component with `magnitude="heteroscedastic"` and `correlation=None`. To use constant
magnitudes, pass `predictors={}` and components with `magnitude="constant"`. The current estimator supports one
heteroscedastic component; manually constructed structures can contain several.

## Grouping and interpolation

GeoUtils returns named predictor index levels and `(value, statistic)` columns. It returns the requested joint
grouping; obtain marginal groupings with separate calls. `observed=False` retains the full grid needed for interpolation.

```python
table = gu.stats.grouped_stats(
    {"error": differences}, {"slope": slope, "curvature": curvature},
    bins={"slope": 10, "curvature": 10}, statistics=[gu.stats.nmad], observed=False,
)
interpolator = xdem.fit.interp_binning(table, value_name="error", statistic="nmad", min_count=100)
prediction = interpolator({"slope": slope, "curvature": curvature})
```

The interpolator accepts a named mapping, fills missing groups and clamps extrapolation to the outer group centres.
It also supports signed bias statistics. `get_perbin_binning` uses the declared intervals without interpolation.
GeoUtils numeric bin edges include the final right edge; explicit IntervalIndexes control closure.

## Variograms

```python
variogram = proxy.variogram(n_pairs=100_000, sampling="loglag", n_runs=3, random_state=42)
fitted = variogram.fit(["gaussian", "spherical"])
fitted.plot()
table = fitted.to_dataframe()
```

`n_pairs` counts sampled pairs directly. The old `subsample` counts observations;
`subsample * (subsample - 1) / 2` provides an approximate pair budget when migrating. Choose GeoUtils `loglag` or
`random_xy` sampling for the new workflow; their random draws can differ from the legacy samplers. Controls such
as `runs`, `samples`, `nb_rings` and `ratio_subsample` remain supported by the deprecated function. When migrating,
choose the corresponding GeoUtils pair sampling options explicitly.

Empirical columns now use `lag`, `semivariance`, `count`, optional `semivariance_error` and bin boundaries. Models
store `effective_range`, `partial_sill`, `nugget` and optional shape parameters. `partial_sill` excludes the nugget.
Keep the GeoUtils model object rather than passing a legacy parameter dataframe between operations. When migrating
plots to GeoUtils, apply options without a direct equivalent to the returned Matplotlib axes. Deprecated xDEM
plotting functions retain their original arguments and rendering.

## Area uncertainty and numerical propagation

```python
standard_errors = xdem.uncertainty.spatial_error_propagation(
    [area_mask], structure, support=dem, predictors=predictors, subsample=1000, random_state=42,
)
result = xdem.uncertainty.propagate_uncertainty(
    dem, "spatial", areas=[area_mask], error_structure=structure, predictors=predictors,
    nsim=100, random_state=42,
)
```

Analytical propagation now evaluates full component covariance at both endpoints instead of multiplying a single
average error map by a representative correlation. Numeric areas retain the stationary circular approximation;
variable magnitudes, independent errors and nuggets require spatial support. `neff_exact` and `neff_hugonnet_approx`
accept an ErrorStructure and optional predictors instead of separate errors and parameter tables.

Numerical propagation returns `PropagationResult`: original `estimate`, ensemble `mean`, propagated `std`, successful
simulation counts, per-output valid counts, and optional `samples`. The initial simulation model is Gaussian.
Keep spatial averaging weights and magnitude predictors fixed, and perform the full derived calculation inside
each realization. Use `operation="terrain"` with an `attribute`, or pass a callable for another calculation.
Aspect uses circular summaries. Coregistration shares this simulation engine and requires an explicit error structure.

Patch estimation retains its existing arguments and dataframe outputs at `xdem.uncertainty.patches_method`.

---
file_format: mystnb
mystnb:
  execution_timeout: 90
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: xdem-env
  language: python
  name: xdem
---
(uncertainty)=

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
```

# Uncertainty analysis

xDEM integrates uncertainty analysis tools from the recent literature that **rely on joint methods from two
scientific fields: spatial statistics and uncertainty quantification**.

While uncertainty analysis technically refers to both systematic and random errors, systematic errors of elevation data
are corrected using {ref}`coregistration` and {ref}`biascorr`, so we here refer to uncertainty analysis for **quantifying and
propagating random errors** (including structured errors).

In detail, xDEM provides tools to:

1. Estimate and model elevation **heteroscedasticity, i.e. variable random errors** (e.g., such as with terrain slope or stereo-correlation),
2. Estimate and model the **spatial correlation of random errors** (e.g., from native spatial resolution or instrument noise),
3. Perform **error propagation to elevation derivatives** (e.g., spatial average, or more complex derivatives such as slope and aspect).

:::{admonition} More reading
:class: tip

For an introduction on spatial statistics applied to uncertainty quantification for elevation data, we recommend reading
the **{ref}`spatial-stats` guide page** and, for details on variography, the **documentation of [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/)**.

Additionally, we recommend reading the **{ref}`static-surfaces` guide page** on which uncertainty analysis relies.
:::


## Quick use

The spatial structure of random elevation errors is represented by an {class}`~xdem.ErrorStructure`. It contains named
independent {class}`~xdem.ErrorComponent` objects, each combining an error magnitude with a normalized spatial
correlation. Keeping those parts separate makes the contribution of short range, long range and independent errors
explicit while supporting random fields and analytical covariance from the same object.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example data and coregistering it"
:  code_prompt_hide: "Hide the code for opening example data and coregistering it"

import xdem
import matplotlib.pyplot as plt
import geoutils as gu
import numpy as np

# Open two DEMs
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
tba_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))

# Open glacier outlines as vector
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# Create a stable ground mask (not glacierized) to mark "inlier data"
inlier_mask = ~glacier_outlines.create_mask(ref_dem)
tba_dem_coreg = tba_dem.coregister_3d(ref_dem, xdem.coreg.NuthKaab() + xdem.coreg.VerticalShift(), inlier_mask=inlier_mask, resample=True)
```

```{code-cell} ipython3
# Estimate the target DEM structure assuming both DEMs have the same error structure
error_structure = tba_dem_coreg.estimate_error_structure(
    ref_dem,
    stable_terrain=inlier_mask,
    other_error="same",
    n_pairs=100_000,
    random_state=42,
)

# Evaluate the combined magnitude from the default slope and curvature predictors
predictors = {
    "slope": tba_dem_coreg.slope(),
    "max_curvature": tba_dem_coreg.max_curvature(),
}
error_magnitude = error_structure.predict_magnitude(predictors, like=tba_dem_coreg)
error_magnitude.plot(cmap="Purples", cbar_title="Elevation error magnitude (m)")

# Inspect representative correlation and generate one spatially structured error realization
print(
    "Errors 1 km apart have a representative correlation of {:.2f} %.".format(
        error_structure.predict_correlation(1000) * 100
    )
)
random_error = error_structure.generate_random_field(tba_dem_coreg, predictors=predictors, random_state=42)
```

The default estimate separates a predictor dependent short range component from a constant long range component.
Custom named components can be passed when the expected instrument or processing errors justify another structure.
The legacy {func}`~xdem.DEM.estimate_uncertainty` tuple interface remains available during the transition.

### Separating components from one error proxy

Estimation first models the total local error magnitude from the predictors. It standardizes the centered error proxy
with that model and fits the requested nested correlations, whose partial sills provide initial component variance
shares. A second fit then compares conditional pair semivariances with the exact endpoint covariance equation. This
uses differences in both distance and local magnitude to distinguish a variable component from fixed contributions.

The separation is conditional on the component forms supplied by the user; one error proxy cannot uniquely identify
arbitrary unknown sources. Overlapping ranges and correlations near the sampled spatial extent are therefore reported
in ``error_structure.fit_diagnostics["identifiability"]``. Pair coordinates and endpoint values are discarded after
their compact conditional summaries have been calculated.

## Summary of available methods

Methods for modelling the structure of error are based on [spatial statistics](https://en.wikipedia.org/wiki/Spatial_statistics), and methods for
propagating errors to spatial derivatives analytically rely on [uncertainty propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty).

To improve the robustness of the uncertainty analysis, we provide refined frameworks for application to elevation data based on
[Rolstad et al. (2009)](http://dx.doi.org/10.3189/002214309789470950) and [Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922),
both for modelling the structure of error and to efficiently perform error propagation.
**These frameworks are generic, simply extending an aspect of the uncertainty analysis to better work on elevation data**,
and thus generally encompass methods described in other studies on the topic (e.g., [Anderson et al. (2019)](http://dx.doi.org/10.1002/esp.4551)).

The tables below summarize the characteristics of these methods.

### Estimating and modeling the structure of error

Frequently, in spatial statistics, a single correlation range is considered ("basic" method below).
However, elevation data often contains errors with correlation ranges spanning different orders of magnitude.
For this, [Rolstad et al. (2009)](http://dx.doi.org/10.3189/002214309789470950) and
[Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922) consider
potential multiple ranges of spatial correlation (instead of a single one). In addition, [Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922)
considers potential heteroscedasticity or variable errors (instead of homoscedasticity, or constant errors), also common in elevation data.

Because accounting for possible multiple correlation ranges also works if you have a single correlation range in your data,
and accounting for potential heteroscedasticity also works on homoscedastic data, **there is little to lose by using
a more advanced framework! (most often, only a bit of additional computation time)**

```{list-table}
   :widths: 1 1 1 1
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * - Method
     - Heteroscedasticity (i.e. variable error)
     - Correlations (single-range)
     - Correlations (multi-range)
   * - Basic
     - ❌
     - ✅
     - ❌
   * - R2009
     - ❌
     - ✅
     - ✅
   * - H2022 (default)
     - ✅
     - ✅
     - ✅
```

For consistency, all methods default to robust estimators: the normalized median absolute deviation (NMAD) for the
spread, and Dowd's estimator for the variogram. See the **{ref}`robust-estimators` guide page** for details.

### Propagating errors to spatial derivatives

Exact uncertainty propagation scales quadratically with data (by computing every pairwise combinations,
for potentially millions of elevation data points or pixels).
To remedy this, [Rolstad et al. (2009)](http://dx.doi.org/10.3189/002214309789470950) and [Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922)
both provide an approximation of exact uncertainty propagations for spatial derivatives (to avoid long
computing times). **These approximations are valid in different contexts**, described below.

```{list-table}
   :widths: 1 1 1 1
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * - Method
     - Accuracy
     - Computing time
     - Validity
   * - Exact discretized
     - Exact
     - Slow on large samples (quadratic complexity)
     - Always
   * - R2009
     - Conservative
     - Instantaneous (numerical integration)
     - Only for near-circular contiguous areas
   * - H2022 (default)
     - Accurate
     - Fast (linear complexity)
     - As long as variance is nearly stationary
```

(spatialstats-heterosc)=

## Core concept for error proxy

Below, we examplify the different steps of uncertainty analysis of **elevation differences between two datasets on
static surfaces as an error proxy**.

In case you want to **convert the uncertainties of elevation differences into that of a "target" elevation dataset**, it can be either assumed that:
- **The "other" elevation dataset is much more precise**, in which case the uncertainties in elevation differences directly approximate that of the "target" elevation dataset,
- **The "other" elevation dataset has similar precision**, in which case the uncertainties of elevation differences quadratically combine twice that of the "target" elevation dataset.

:::{admonition} More reading (reminder)
:class: tip

To clarify these conversions of error proxy, see the **{ref}`static-surfaces` guide page**.
For more statistical background on the methods below, see the **{ref}`spatial-stats` guide page**.
:::

(error-struc)=
## Spatial structure of error

An {class}`~xdem.ErrorStructure` estimates error magnitude and spatial correlation together, then stores them as
separate parts of named components. This is important when the same error proxy contains, for example, a variable
short range contribution and a constant long range contribution: each contribution retains its own magnitude and
correlation rather than being summarized by a single error map and variogram.

### Magnitude and correlation

Use {meth}`~xdem.ErrorStructure.estimate` when an elevation difference on stable terrain is already available.
For two elevation datasets, {func}`~xdem.uncertainty.estimate_error_structure` also prepares their common finite
support through GeoUtils cosampling and applies the selected error attribution.

Grouped magnitude estimation uses `geoutils.stats.grouped_stats`. Its tables have named predictor index levels and
`(value, statistic)` columns. {func}`~xdem.fit.interp_binning` interpolates these groups, with missing groups filled
and predictions clamped to the outer group centres. {class}`~xdem.ErrorMagnitude` adds the error-specific scaling,
variance subtraction and positivity constraints. Correlations are portable `geoutils.VariogramModel` objects.

```python
# Inspect the fitted components without keeping the sampled pairs
error_structure.plot_magnitude()
error_structure.plot_correlation()

# Inspect a component's magnitude model and normalized correlation
short_range = error_structure["short_range"]
print(short_range.magnitude.grouped_statistics)
print(short_range.correlation)
```

For generic variography outside uncertainty estimation, use `raster.variogram(...)` or `pointcloud.variogram(...)`,
then `.fit(...)`, `.plot()` and `.to_dataframe()` on the resulting GeoUtils object.

## Propagation of errors

The same error structure supports analytical covariance propagation and numerical propagation through arbitrary
calculations. In both cases the component magnitudes are evaluated on the observation support.

### Analytical spatial averages

{func}`~xdem.uncertainty.spatial_error_propagation` returns the standard uncertainty of each area average. With
normalized averaging weights, it sums the covariance between observations, evaluating each component's magnitude
at both endpoints. This accounts for components with different spatial scales and different magnitude predictors.

```python
outline_brom = gu.Vector(glacier_outlines.ds[glacier_outlines.ds["NAME"] == "Brombreen"])
standard_errors = xdem.uncertainty.spatial_error_propagation(
    [outline_brom], error_structure, support=tba_dem_coreg, predictors=predictors,
    subsample=1000, random_state=42,
)
```

Pass `subsample=None` for the complete discrete covariance sum. A bare numeric area retains the stationary circular
approximation; use a spatial mask or vector and explicit support for variable magnitudes or independent errors.
{func}`~xdem.uncertainty.number_effective_samples` uses the squared average local standard deviation divided by the
variance of the average, preserving the relation between these two uncertainty summaries.

### Numerical spatial averages, terrain and coregistration

{func}`~xdem.uncertainty.propagate_uncertainty` generates Gaussian elevation error fields from the supplied structure,
adds each field to the original elevations and reruns a calculation. It shares one simulation engine between
spatial, terrain, coregistration and user supplied calculations. Magnitude predictors remain fixed during simulation.

```python
# Average elevations inside a vector area for every realization
area_result = xdem.uncertainty.propagate_uncertainty(
    tba_dem_coreg, "spatial", areas=[outline_brom], error_structure=error_structure,
    predictors=predictors, nsim=100, random_state=42,
)

# Recompute slope for every realization
slope_result = xdem.uncertainty.propagate_uncertainty(
    tba_dem_coreg, "terrain", attribute="slope", error_structure=error_structure,
    predictors=predictors, nsim=100, random_state=42,
)
slope_result.std.plot(cbar_title="Slope standard uncertainty (degrees)")

# A callable can describe a complete calculation, including several successive operations
mask_brom = outline_brom.create_mask(tba_dem_coreg, as_array=True)
mean_slope_result = xdem.uncertainty.propagate_uncertainty(
    tba_dem_coreg,
    lambda dem: np.ma.mean(xdem.terrain.slope(dem).data[mask_brom]),
    error_structure=error_structure, predictors=predictors, nsim=100, random_state=42,
)
```

The returned {class}`~xdem.uncertainty.PropagationResult` contains the original `estimate`, ensemble `mean` and `std`,
successful simulation counts and per-output valid counts. Spatial results retain their coordinates and masks.
For nonlinear operations the ensemble mean can differ from the original estimate. Terrain aspect automatically
uses circular summaries; custom angular calculations can pass `circular_period`.

Individual outputs are retained only with `return_samples=True`. By default errors in a user supplied operation
are raised; `on_error="warn"` records and skips failed realizations. At least two successful realizations are required.

Coregistration uses fresh fitted objects for every realization and records failed fits. The convenience function
{func}`~xdem.uncertainty.propagate_uncertainty_coreg` returns the mean/STD table, the simulation table and fitted
objects. The general entry point with `operation="coreg"` returns a `PropagationResult` with those diagnostics in
its metadata. `precoreg=True` performs an initial alignment before fitting the simulated residual transforms.

### Empirical patch estimates

{func}`~xdem.uncertainty.patches_method` estimates uncertainty from observed proxy values in many patches. It retains
the convolution and quadrant algorithms, the valid-area threshold, and the reported rasterized patch areas.
It operates on observations directly and does not require an ErrorStructure.

### Migrating existing code

All functions in `xdem.spatialstats` are deprecated. The [migration guide](uncertainty_migration.md) maps the former
functions to GeoUtils, `xdem.fit` and the uncertainty modules, with examples of the new input and output contracts.

---
file_format: mystnb
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
are corrected using {ref}`coregistration` and {ref}`biascorr`, so we here refer to **uncertainty analysis for quantifying and
propagating random errors (including structured errors)**.

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

The estimation of the spatial structure of random errors of elevation data is conveniently
wrapped in a single method {func}`~xdem.DEM.estimate_uncertainty`, for which the steps are detailed below.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example data and coregistering it"
:  code_prompt_hide: "Hide the code for opening example data and coregistering it"

import xdem
import matplotlib.pyplot as plt
import geoutils as gu

# Open two DEMs
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
tba_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))

# Open glacier outlines as vector
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# Create a stable ground mask (not glacierized) to mark "inlier data"
inlier_mask = ~glacier_outlines.create_mask(ref_dem)
tba_dem_coreg = tba_dem.coregister_3d(ref_dem, inlier_mask=inlier_mask)
```

```{code-cell} ipython3
# Estimate elevation uncertainty assuming both DEMs have similar precision
sig_dem, rho_sig = tba_dem_coreg.estimate_uncertainty(ref_dem, stable_terrain=inlier_mask, precision_of_other="same")

# The error map variability is estimated from slope and curvature by default
sig_dem.plot(cmap="Purples", cbar_title=r"Error in elevation (1$\sigma$, m)")

# The spatial correlation function represents how much errors are correlated at a certain distance
print("Random elevation errors at a distance of 1 km are correlated at {:.2f} %.".format(rho_sig(1000) * 100))
```

## Summary of available methods

Methods for modelling the structure of error are based on [spatial statistics](https://en.wikipedia.org/wiki/Spatial_statistics), and methods for 
propagating errors to spatial derivatives analytically rely on [uncertainty propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty).

To improve the robustness of the uncertainty analysis, we provide refined frameworks for application to elevation data based on 
[Rolstad et al. (2009)](http://dx.doi.org/10.3189/002214309789470950) and [Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922), 
both for modelling the structure of error and to efficiently perform error propagation.
**These frameworks are generic, simply extending an aspect of the uncertainty analysis to better work on elevation data**, 
and thus generally encompass methods described in other studies on the topic (e.g., [Anderson et al. (2019)](http://dx.doi.org/10.1002/esp.4551)).

The tables below summarize the characteristics of these methods.

### Estimating and modelling the structure of error

Traditionally, in spatial statistics, a single correlation range is considered ("traditional" method below). 
However, elevation data often contains errors with correlation ranges spanning different orders of magnitude.
For this, [Rolstad et al. (2009)](http://dx.doi.org/10.3189/002214309789470950) and 
[Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922) considers 
potential multiple ranges of spatial correlation (instead of a single one). In addition, [Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922) 
considers potential heteroscedasticity or variable errors (instead of homoscedasticity, or constant errors), also common in elevation data.

Because accounting for possible multiple correlation ranges also works if you have a single correlation range in your data, 
and accounting for potential heteroscedasticity also works on homoscedastic data, **there is little to lose by using 
a more advanced framework! (most often, only a bit of additional computation time)**

```{list-table}
   :widths: 1 1 1 1 1
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * - Method
     - Heteroscedasticity (i.e. variable error)
     - Correlations (single-range)
     - Correlations (multi-range)
     - Outlier-robust
   * - Traditional
     - ❌
     - ✅
     - ❌
     - ❌
   * - R2009
     - ❌
     - ✅
     - ✅
     - ❌
   * - H2022 (default)
     - ✅
     - ✅
     - ✅
     - ✅
```

### Propagating errors to spatial derivatives

Exact uncertainty propagation scales exponentially with data (by computing every pairwise combinations, 
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
     - Slow on large samples (exponential complexity)
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

Below, we examplify the different steps of uncertainty analysis relying on **elevation differences between two datasets on 
static surfaces as an error proxy**.

To convert into the uncertainty of one elevation datasets, it is either assumed that the other dataset is much more 
precise, or that they have similar precision.

:::{admonition} More reading (reminder)
:class: tip

To clarify these error proxy aspects, see the **{ref}`static-surfaces` guide page**.
For more statistical background on the methods below, see the **{ref}`spatial-stats` guide page**.
:::


## Spatial structure of error

Below we detail the steps used to estimate heteroscedasticity and spatial correlation of errors in
{func}`~xdem.DEM.estimate_uncertainty`, which are most easily customized by calling subfunctions independently.

### Heteroscedasticity

Elevation [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity) (or variability in
random elevation errors) can be empirically estimated by [data binning](https://en.wikipedia.org/wiki/Data_binning) 
in N-dimensions with the function {func}`xdem.spatialstats.nd_binning`:

The most common explanatory variables for elevation heteroscedasticity are:

> - the terrain slope and terrain curvature (see {ref}`terrain-attributes`),
> - the quality of stereo-correlation (stereo DEMs),
> - the interferometric coherence ([InSAR](https://en.wikipedia.org/wiki/Interferometric_synthetic-aperture_radar) DEMs).

```{code-cell} ipython3
# Derive slope and curvature
slope = xdem.terrain.get_terrain_attribute(ref_dem, attribute=["slope", "curvature"])

# Estimate the measurement error by bin of slope and curvature
df_h = xdem.spatialstats.nd_binning(
    dh_arr, list_var=[slope_arr], list_var_names=["slope"], statistics=["count", xdem.spatialstats.nmad]
)
```

Once estimated per bin, elevation heteroscedasticity can be modelled either by a function fit, or numerically by 
N-D linear interpolation using {func}`xdem.spatialstats.interp_nd_binning`:

```{code-cell} ipython3
# Derive a numerical function of the measurement error
err_dh = xdem.spatialstats.interp_nd_binning(df_h, list_var_names=["slope", "curvature"])
```

Which can be used to derive the estimated random error for any slope and curvature, and yield a map of elevation 
change errors that, depending on your assumption can translate

### Spatial correlation of errors

If heteroscedasticity was considered, elevation differences can be standardized to improve the estimation 
of spatial correlations:

```{code-cell} ipython3
# Standardize the data
z_dh = dh_arr / err_dh(slope_arr)
```

Then, an empirical variogram describing

To remedy this issue, xDEM provides {func}`xdem.spatialstats.sample_empirical_variogram`, an empirical variogram estimation tool
that encapsulates a pairwise subsampling method described in `skgstat.MetricSpace.RasterEquidistantMetricSpace`.
This method compares pairwise distances between a center subset and equidistant subsets iteratively across a grid, based on
[sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix) routines computing pairwise distances of two separate
subsets, as in [scipy.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
(instead of using pairwise distances within the same subset, as implemented in most spatial statistics packages).
The resulting pairwise differences are evenly distributed across the grid and across lag classes (in 2 dimensions, this
means that lag classes separated by a factor of $\sqrt{2}$ have an equal number of pairwise differences computed).

```{code-cell} ipython3
# Sample empirical variogram
df_vgm = xdem.spatialstats.sample_empirical_variogram(values=dh, subsample=10, random_state=42)
```

The variogram is returned as a {class}`~pandas.DataFrame` object.

With all spatial lags sampled evenly, estimating a variogram requires significantly less samples, increasing the
robustness of the spatial correlation estimation and decreasing computing time!

Once an empirical variogram is estimated, fitting a function model allows to simplify later analysis by directly
providing a function form (e.g., for kriging equations, or uncertainty analysis - see {ref}`spatialstats-errorpropag`),
which would otherwise have to be numerically modelled.

Generally, in spatial statistics, a single model is used to describe the correlation in the data.
In elevation data, however, spatial correlations are observed at different scales, which requires fitting a sum of models at
multiple ranges (introduced in [Rolstad et al. (2009)](https://doi.org/10.3189/002214309789470950) for glaciology
applications).

This can be performed through the function {func}`xdem.spatialstats.fit_sum_model_variogram`, which expects as input a
`pd.Dataframe` variogram.

```{code-cell} ipython3
# Fit sum of double-range spherical model
func_sum_vgm, params_variogram_model = xdem.spatialstats.fit_sum_model_variogram(
    list_models=["Gaussian", "Spherical"], empirical_variogram=df_vgm
)
```


## Propagation of errors

### Spatial derivatives

After quantifying and modelling spatial correlations, those an effective sample size, and elevation measurement error:

```{code-cell} ipython3
# Calculate the area-averaged uncertainty with these models
# neff = xdem.spatialstats.number_effective_samples(area=1000, params_variogram_model=params_variogram_model)
```

### Other derivatives

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

xDEM integrates spatial uncertainty analysis tools from the recent literature that **rely on joint methods from two
scientific fields: spatial statistics and uncertainty quantification**.

While uncertainty analysis technically refers to both systematic and random errors, systematic errors of elevation data
are corrected using {ref}`coregistration` and {ref}`biascorr`, so we here refer to **uncertainty analysis for quantifying and
propagating random errors**.

In detail, we provide tools to:

1. Account for elevation **heteroscedasticity** (e.g., varying precision such as with terrain slope or stereo-correlation),
2. Quantify the **spatial correlation of random errors** (e.g., from native spatial resolution or instrument noise),
3. Perform an **error propagation to elevation derivatives** (e.g., spatial average, or more complex derivatives such as slope and aspect).

:::{admonition} More reading
:class: tip

For an introduction on spatial statistics applied to uncertainty quantification for elevation data, we recommend reading
the **{ref}`spatial-stats` guide page** and, for details on variography, the **documentation of [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/)**.

Additionally, we recommend reading the **{ref}`static-surfaces` guide page** on which uncertainty analysis relies.
:::

## Quick use

The estimation of the spatial structure of random errors of elevation data (heteroscedas) is conveniently
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

Our methods for modelling the structure of error in DEMs and propagating errors to spatial derivatives analytically
are primarily based on [Rolstad et al. (2009)]() and [Hugonnet et al. (2022)]().

These frameworks are generic and thus encompass that of most other studies on the topic (e.g., Anderson et al. (2020),
others), referred to as "traditional" below. This is because accounting for possible multiple correlation ranges also
works for the case of single correlation range, or accounting for potential heteroscedasticity also works on
homoscedastic elevation data.

The tables below summarize the characteristics of these three category of methods.

### Estimating and modelling the structure of error

```{list-table}
   :widths: 1 1 1 1 1
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * - Method
     - Heteroscedasticity
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

```{list-table}
   :widths: 1 1 1 1
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * - Method
     - Accuracy
     - Computing time
     - Remarks
   * - Exact discretized
     - Exact
     - Slow on large samples
     - Complexity scales exponentially
   * - R2009
     - Conservative
     - Instantaneous
     - Only valid for near-circular contiguous areas
   * - H2022 (default)
     - Accurate
     - Fast
     - Complexity scales linearly
```

(spatialstats-heterosc)=

## Spatial structure of error

Below we detail the steps used to estimate heteroscedasticity and spatial correlation of errors in
{func}`~xdem.DEM.estimate_uncertainty`, which are most easily customized by calling subfunctions independently.

### Elevation heteroscedasticity

Elevation [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity) corresponds to a variability in
precision (random errors) of elevation data, that is often linked to terrain, instrument or processing errors.

$$
\sigma_{h} = \sigma_{h}(\textrm{var}_{1},\textrm{var}_{2}, \textrm{...}) \neq \textrm{constant}
$$

Owing to the large number of samples of elevation data, we can easily estimate this variability by
[binning](https://en.wikipedia.org/wiki/Data_binning) the data and estimating the statistical dispersion (see
{ref}`robuststats-meanstd`) across several explanatory variables using {func}`xdem.spatialstats.nd_binning`.

```{code-cell} ipython3
:tags: [hide-input, hide-output]
import geoutils as gu
import numpy as np

import xdem

# Load data
dh = gu.Raster(xdem.examples.get_path("longyearbyen_ddem"))
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
glacier_mask = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(dh)

slope = xdem.terrain.get_terrain_attribute(ref_dem, attribute=["slope"])

# Keep only stable terrain data
dh.load()
dh.set_mask(mask)
dh_arr = gu.raster.get_array_and_mask(dh)[0]
slope_arr = gu.raster.get_array_and_mask(slope)[0]

# Subsample to run the snipped code faster
indices = gu.raster.subsample_array(dh_arr, subsample=10000, return_indices=True, random_state=42)
dh_arr = dh_arr[indices]
slope_arr = slope_arr[indices]
```

```{code-cell} ipython3
# Estimate the measurement error by bin of slope, using the NMAD as robust estimator
df_ns = xdem.spatialstats.nd_binning(
    dh_arr, list_var=[slope_arr], list_var_names=["slope"], statistics=["count", xdem.spatialstats.nmad]
)
```

The most common explanatory variables are the terrain slope, terrain curvature, quality of stereo-correlation and

> - the terrain slope and terrain curvature (see {ref}`terrain-attributes`) that can explain a large part of the terrain-related variability in error,
> - the quality of stereo-correlation that can explain a large part of the measurement error of DEMs generated by stereophotogrammetry,
> - the interferometric coherence that can explain a large part of the measurement error of DEMs generated by [InSAR](https://en.wikipedia.org/wiki/Interferometric_synthetic-aperture_radar).

Once quantified, elevation heteroscedasticity can be modelled numerically by linear interpolation across several
variables using {func}`xdem.spatialstats.interp_nd_binning`.

```{code-cell} ipython3
# Derive a numerical function of the measurement error
err_dh = xdem.spatialstats.interp_nd_binning(df_ns, list_var_names=["slope"])
```

### Standardization

```{code-cell} ipython3
# Standardize the data
z_dh = dh_arr / err_dh(slope_arr)
```

### Spatial correlation of errors

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

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

(biascorr)=

# Bias correction

In xDEM, bias-correction methods correspond to **transformations that cannot be described as a 3-dimensional
affine function** (see {ref}`coregistration`), and aim at correcting both systematic elevation errors and
spatially-structured random errors.

Contrary to affine coregistration methods, bias corrections are **not limited to the information in the elevation data**. They can be
passed any external variables (e.g., land cover type, processing metric) to attempt to identify and correct biases.
Still, many methods rely either on coordinates (e.g., deramping, along-track corrections) or terrain
(e.g., curvature- or elevation-dependant corrections), derived solely from the elevation data.

```{code-cell} ipython3
:tags: [remove-cell]

#################################################################################
# This a temporary trick to allow vertical referencing to work in other notebooks
#################################################################################
# Somehow, only on Readthedocs (locally works fine), the first notebook to run (in alphabetical order) fails
# to download from PROJ... while all other notebook render normally.
# The first alphabetical notebook is "biascorr", so adding this trick here

import xdem
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
# Define the vertical CRS as the 3D ellipsoid of the 2D CRS
dem.set_vcrs("Ellipsoid")
# Transform to the EGM96 geoid
dem.to_vcrs("EGM96")
```

## Quick use

Bias-correction methods are **used the same way as coregistrations**:

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
```

```{code-cell} ipython3
import xdem
import numpy as np

# Create a bias-correction
biascorr = xdem.coreg.DirectionalBias(angle=45, fit_or_bin="bin", bin_sizes=200, bin_apply_method="per_bin", bin_statistic=np.mean)
```

Bias correction can estimate and correct the bias **by a parametric fit** using `fit_or_bin="fit"` linked to  `fit_` parameters, **by applying
a binned statistic** using `fit_or_bin="bin"` linked to `bin_` parameters, or **by a parametric fit on the binned data** using `fit_or_bin="bin_and_fit"`
linked to all parameters.

Predefined bias corrections usually take additional arguments such as `angle` for {class}`~xdem.coreg.DirectionalBias`,
`poly_order` for {class}`~xdem.coreg.Deramp` and `attribute` for {class}`~xdem.coreg.TerrainBias`.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

# Open a reference and to-be-aligned DEM
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
tba_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
```

Once defined, they can be applied the same two ways as for coregistration (using {func}`~xdem.coreg.Coreg.fit` and
{func}`~xdem.coreg.Coreg.apply` separately allows to re-apply the same correction to different elevation data).

```{code-cell} ipython3
# Coregister with bias correction by calling the DEM method
corrected_dem = tba_dem.coregister_3d(ref_dem, biascorr)
# (Equivalent) Or by calling the fit and apply steps
corrected_dem = biascorr.fit_and_apply(ref_dem, tba_dem)
```

Additionally, **bias corrections can be customized to use any number of variables to correct simultaneously**,
by defining `bias_var_names` in {class}`~xdem.coreg.BiasCorr`  and passing a `bias_vars` dictionary arrays or rasters
to {func}`~xdem.coreg.Coreg.fit` and {func}`~xdem.coreg.Coreg.apply`. See {ref}`custom-biascorr` for more details.


## The modular {class}`~xdem.coreg.BiasCorr` object

### Inherited from {class}`~xdem.coreg.Coreg`

Each bias-correction method in xDEM inherits their interface from the {class}`~xdem.coreg.Coreg` class (see {ref}`coreg_object`).
This implies that bias-correction methods can be combined in a {class}`~xdem.coreg.CoregPipeline` with any other methods, or
applied in a block-wise manner through {class}`~xdem.coreg.BlockwiseCoreg`.

**Inheritance diagram of co-registration and bias corrections:**

```{eval-rst}
.. inheritance-diagram:: xdem.coreg.base.Coreg xdem.coreg.affine xdem.coreg.biascorr
        :top-classes: xdem.coreg.Coreg
```

The main difference with {class}`~xdem.coreg.Coreg` is that a {class}`~xdem.coreg.BiasCorr` has a new `bias_var_names`
argument which allows declaring the names of N bias-correction variables that will be passed, which **corresponds to the
number of simultaneous dimensions in which the bias correction is performed**.
This step is implicit for predefined methods such as {class}`~xdem.coreg.DirectionalBias`.

### Modular estimation

Bias-correction methods have three ways of estimating and correcting a bias in N-dimensions:

- **Performing a binning of the data** along variables with a statistic (e.g., median), then applying the statistics in each bin,
- **Fitting a parametric function** to the variables, then applying that function,
- **(Recommended<sup>1</sup>) Fitting a parametric function on a data binning** of the variable, then applying that function.

```{margin}
<sup>1</sup>DEM correction is a big data problem plagued by outliers, more robust and computationally efficient when binning with robust estimators.

See the **{ref}`robust-estimators` guide page** for more details.
```

The parameters related to fitting or binning are the same for every {func}`~xdem.coreg.BiasCorr` method:

- `fit_or_bin` to either fit a parametric model to the bias by passing **"fit"**, perform an empirical binning of the bias by passing **"bin"**, or to fit a parametric model to the binning with **"bin_and_fit" (recommended)**,
- `fit_func` to pass any parametric function to fit to the bias,
- `fit_optimizer` to pass any optimizer function to perform the fit minimization,
- `bin_sizes` to pass the size or edges of the bins for each variable,
- `bin_statistic` to pass the statistic to compute in each bin,
- `bin_apply_method` to pass the method to apply the binning for correction.

For predefined methods, the default values of these parameters differ. For instance, a {class}`~xdem.coreg.Deramp` generally performs well
with a **"fit"** estimation on a subsample, and thus has a fixed `fit_func` (2D polynomial) solved by the classic optimizer {func}`scipy.optimize.curve_fit`.
In contrast, a {class}`~xdem.coreg.TerrainBias` is generally hard to model parametrically, and thus defaults to a **"bin"** estimation.

Finally, each bias-correction approach has the following methods:

- {func}`~xdem.coreg.Coreg.fit` for estimating the bias, which expects a new `bias_vars` dictionary **except for predefined methods** such as {class}`~xdem.coreg.DirectionalBias`,
- {func}`~xdem.coreg.Coreg.apply` for correcting the bias on a DEM, which also expects a `bias_vars` dictionary **except for predefined methods**.

### Good practices

Several good practices help performing a successful bias correction:

- **Avoid using "fit" with a subsample size larger than 1,000,000:** Otherwise the optimizer will be extremely slow and might fail with a memory error; consider using "bin_and_fit" instead to reduce the data size before the optimization which still allows to utilize all the data,
- **Avoid using "fit" or "bin_and_fit" for more than 2 dimensions (input variables):** Fitting a parametric form in more than 2 dimensions is quite delicate, consider using "bin" or sequential 1D corrections instead,
- **Use at least 1000 bins for all dimensions, being mindful about dimension number:** Using a small bin size is generally too rough, but a large bin size will grow exponentially with the number of bias variables,
- **Use customized bin edges for data with large extreme values:** Passing simply a bin size will set the min/max of the data as the full binning range, which can be impractical (e.g., most curvatures lie between -2/2 but can take values of 10000+).

## Bias-correction methods

```{important}
Below we **create biased elevation data to examplify the different methods** in relation to their type of correction.

See bias correction on real data in the **{ref}`examples-basic` and {ref}`examples-advanced` gallery examples**!
```

(deramp)=
### Deramping

{class}`xdem.coreg.Deramp`

- **Performs:** Correction with a 2D polynomial of degree N.
- **Supports weights:** Yes.
- **Pros:** Can help correct a large category of biases (lens deformations, camera positioning), and runs fast.
- **Cons:** Overfits with limited static surfaces.

Deramping works by estimating and correcting for an N-degree polynomial over the entire elevation difference.


```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for adding a ramp bias"
:  code_prompt_hide: "Hide the code for adding a ramp bias"

# Get grid coordinates
xx, yy = np.meshgrid(np.arange(0, ref_dem.shape[1]), np.arange(0, ref_dem.shape[0]))

# Create a ramp bias and add to the DEM
cx = ref_dem.shape[1] / 2
cy = ref_dem.shape[0] / 2
synthetic_bias = 20 * ((xx - cx)**2 + (yy - cy)**2) / (cx * cy)
synthetic_bias -= np.median(synthetic_bias)
tbc_dem_ramp = ref_dem + synthetic_bias
```

```{code-cell} ipython3
# Instantiate a 2nd order 2D deramping
deramp = xdem.coreg.Deramp(poly_order=2)
# Fit and apply
corrected_dem = deramp.fit_and_apply(ref_dem, tbc_dem_ramp)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before deramp")
(tbc_dem_ramp - ref_dem).plot(cmap='RdYlBu', ax=ax[0])
ax[1].set_title("After deramp")
(corrected_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

### Directional biases

{class}`xdem.coreg.DirectionalBias`

- **Performs:** Correct biases along a direction.
- **Supports weights:** Yes.
- **Pros:** Correcting undulations or jitter, common in both stereo and radar DEMs, or strips common in scanned imagery.
- **Cons:** Long optimization when fitting a sum of sinusoids.

For strip-like errors, performing an empirical correction using only a binning with `fit_or_bin="bin"` allows more
flexibility than a parametric form, but requires a large amount of static surfaces.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for adding a strip bias"
:  code_prompt_hide: "Hide the code for adding a strip bias"

# Get rotated coordinates along an angle
angle = 60
xx = gu.raster.get_xy_rotated(ref_dem, along_track_angle=angle)[0]

# Create a strip bias and add to the DEM
synthetic_bias = np.zeros(np.shape(ref_dem.data))
xmin = np.min(xx)
synthetic_bias[np.logical_and((xx - xmin)<1200, (xx - xmin)>800)] = 20
synthetic_bias[np.logical_and((xx - xmin)<2800, (xx - xmin)>2500)] = -10
synthetic_bias[np.logical_and((xx - xmin)<5300, (xx - xmin)>5000)] = 10
synthetic_bias[np.logical_and((xx - xmin)<15000, (xx - xmin)>14500)] = 5
synthetic_bias[np.logical_and((xx - xmin)<21000, (xx - xmin)>20000)] = -15
tbc_dem_strip = ref_dem + synthetic_bias
```

```{code-cell} ipython3
# Define a directional bias correction at a certain angle (degrees), for a binning of 1000 bins
dirbias = xdem.coreg.DirectionalBias(angle=60, fit_or_bin="bin", bin_sizes=1000)
# Fit and apply
corrected_dem = dirbias.fit_and_apply(ref_dem, tbc_dem_strip)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before directional\nde-biasing")
(tbc_dem_strip - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After directional\nde-biasing")
(corrected_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

### Terrain biases

{class}`xdem.coreg.TerrainBias`

- **Performs:** Correct biases along a terrain attribute.
- **Supports weights:** Yes.
- **Pros:** Useful to correct for instance curvature-related bias due to different native resolution between elevation data.
- **Cons:** For curvature-related biases, only works for elevation data with relatively close native resolution.

The default optimizer for terrain biases optimizes a 1D polynomial with an order from 1 to 6,
and keeps the best performing fit.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for adding a curvature bias"
:  code_prompt_hide: "Hide the code for adding a curvature bias"

# Get maximum curvature
maxc = ref_dem.maximum_curvature()

# Create a bias depending on bins
synthetic_bias = np.zeros(np.shape(ref_dem.data))

# For each bin, add the curvature bias
bin_edges = np.array((-1, -0.5, -0.1, 0.1, 0.5, 2, 5))
bias_per_bin = np.array((-10, -5, 0, 5, 10, 20))
for i in range(len(bin_edges) - 1):
    synthetic_bias[np.logical_and(maxc.data >= bin_edges[i], maxc.data < bin_edges[i + 1])] = bias_per_bin[i]
tbc_dem_curv = ref_dem + synthetic_bias
```

```{code-cell} ipython3
# Instantiate a 1st order terrain bias correction for curvature
terbias = xdem.coreg.TerrainBias(terrain_attribute="maximum_curvature",
                                 bin_sizes={"maximum_curvature": np.linspace(-5, 5, 1000)},
                                 bin_apply_method="per_bin")

# We have to pass the original curvature here
corrected_dem = terbias.fit_and_apply(ref_dem, tbc_dem_curv, bias_vars={"maximum_curvature": maxc})
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before terrain\nde-biasing")
(tbc_dem_curv - ref_dem).plot(cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[0])
ax[1].set_title("After terrain\nde-biasing")
(corrected_dem - ref_dem).plot(cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```


(custom-biascorr)=
### Using custom variables

All bias-corrections methods are inherited from generic classes that perform corrections in 1-, 2- or N-D. Having these
separate helps the user navigating the dimensionality of the functions, optimizer, binning or variables used.

{class}`xdem.coreg.BiasCorr`

- **Performs:** Correct biases with any function and optimizer, or any binning, in 1-, 2- or N-D.
- **Supports weights:** Yes.
- **Pros:** Versatile.
- **Cons:** Needs more setting up!


```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code for creating inlier mask and coregistration"
:  code_prompt_hide: "Hide code for creating inlier mask and coregistration"

import geoutils as gu

# Open glacier outlines as vector
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# Create a stable ground mask (not glacierized) to mark "inlier data"
inlier_mask = ~glacier_outlines.create_mask(ref_dem)

# We align the two DEMs before doing any bias correction
tba_dem_nk = tba_dem.coregister_3d(ref_dem, xdem.coreg.NuthKaab(), resample=True)
```

```{code-cell} ipython3
# Create a bias correction defining three custom variable names that will be passed later
# We force a binning method, more simple in 3D
biascorr = xdem.coreg.BiasCorr(bias_var_names=["aspect", "slope", "elevation"], fit_or_bin="bin", bin_sizes=5)

# Derive curvature and slope
aspect, slope = ref_dem.get_terrain_attribute(["aspect", "slope"])

# Pass the variables to the fit_and_apply function matching the names declared above
corrected_dem = biascorr.fit_and_apply(
    ref_dem,
    tba_dem_nk,
    inlier_mask=inlier_mask,
    bias_vars={"aspect": aspect, "slope": slope, "elevation": ref_dem}
)
```

```{warning}
Using any custom variables, and especially in many dimensions, **can lead to over-correction and introduce new errors**.
For instance, elevation-dependent corrections (as shown below) typically introduce new errors (due to more high curvatures
at high elevation such as peaks, and low curvatures at low elevation with flat terrain).

For this reason, it is important to check the sanity of elevation differences after correction!
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before 3D\nde-biasing")
(tba_dem_nk - ref_dem).plot(cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[0])
ax[1].set_title("After 3D\nde-biasing")
(corrected_dem - ref_dem).plot(cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

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

(biascorr)=

# Bias correction

In xDEM, bias-correction methods correspond to non-rigid transformations that cannot be described as a 3-dimensional
affine function (see {ref}`coregistration`).

Contrary to rigid coregistration methods, bias corrections are not limited to the information in the DEMs. They can be
passed any external variables (e.g., land cover type, processing metric) to attempt to identify and correct biases in
the DEM. Still, many methods rely either on coordinates (e.g., deramping, along-track corrections) or terrain
(e.g., curvature- or elevation-dependant corrections), derived solely from the DEM.

## The {class}`~xdem.BiasCorr` object

Each bias-correction method in xDEM inherits their interface from the {class}`~xdem.Coreg` class (see {ref}`coreg_object`).
This implies that bias-correction methods can be combined in a {class}`~xdem.CoregPipeline` with any other methods, or
applied in a block-wise manner through {class}`~xdem.BlockwiseCoreg`.

**Inheritance diagram of co-registration and bias corrections:**

```{eval-rst}
.. inheritance-diagram:: xdem.coreg xdem.biascorr
        :top-classes: xdem.Coreg
```

As a result, each bias-correction approach has the following methods:

- {func}`~xdem.BiasCorr.fit` for estimating the bias.
- {func}`~xdem.BiasCorr.apply` for correcting the bias on a DEM.

## Modular estimators

Bias-correction methods have 3 main ways of estimating and correcting a bias, both relying on one or several variables:

- **Performing a binning of the data** along variables with a statistic (e.g., median), and applying the statistics in each bin,
- **Fitting a parametric function** to the variables, and applying that function,
- **(Recommended<sup>1</sup>) Fitting a parametric function on a data binning** of the variable, and applying that function.

```{margin}
<sup>1</sup>DEM alignment is a big data problem often plagued by outliers, greatly **simplified** and **accelerated** by binning with robust estimators.
```

To define the parameters related to fitting and/or binning, every {func}`~xdem.BiasCorr` is instantiated with the same arguments:

- `fit_or_bin` to either fit a parametric model to the bias by passing "fit", perform an empirical binning of the bias by passing "bin", or to fit a parametric model to the binning with "bin_and_fit" **(recommended)**,
- `fit_func` to pass any parametric function to fit to the bias,
- `fit_optimizer` to pass any optimizer function to perform the fit minimization,
- `bin_sizes` to pass the size or edges of the bins for each variable,
- `bin_statistic` to pass the statistic to compute in each bin,
- `bin_apply_method` to pass the method to apply the binning for correction.

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import geoutils as gu
import numpy as np

import xdem

# Open a reference DEM from 2009
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
# Open a to-be-aligned DEM from 1990
tba_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem")).reproject(ref_dem, silent=True)

# Open glacier polygons from 1990, corresponding to unstable ground
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
# Create an inlier mask of terrain outside the glacier polygons
inlier_mask = glacier_outlines.create_mask(ref_dem)
```

(biascorr-deramp)=

## Deramping

{class}`xdem.biascorr.Deramp`

- **Performs:** Correct biases with a 2D polynomial of degree N.
- **Supports weights** Yes.
- **Recommended for:** Residuals from camera model.

Deramping works by estimating and correcting for an N-degree polynomial over the entire dDEM between a reference and the DEM to be aligned.
This may be useful for correcting small rotations in the dataset, or nonlinear errors that for example often occur in structure-from-motion derived optical DEMs (e.g. Rosnell and Honkavaara [2012](https://doi.org/10.3390/s120100453); Javernick et al. [2014](https://doi.org/10.1016/j.geomorph.2014.01.006); Girod et al. [2017](https://doi.org/10.5194/tc-11827-2017)).

### Limitations

Deramping does not account for horizontal (X/Y) shifts, and should most often be used in conjunction with other methods.

1st order deramping is not perfectly equivalent to a rotational correction: values are simply corrected in the vertical direction, and therefore includes a horizontal scaling factor, if it would be expressed as a transformation matrix.
For large rotational corrections, [ICP] is recommended.

### Example

```{code-cell} ipython3
from xdem import biascorr

# Instantiate a 1st order deramping
deramp = biascorr.Deramp(poly_order=1)
# Fit the data to a suitable polynomial solution
deramp.fit(ref_dem, tba_dem, inlier_mask=inlier_mask)

# Apply the transformation
corrected_dem = deramp.apply(tba_dem)
```

## Directional biases

{class}`xdem.biascorr.DirectionalBias`

- **Performs:** Correct biases along a direction of the DEM.
- **Supports weights** Yes.
- **Recommended for:** Undulations or jitter, common in both stereo and radar DEMs.

The default optimizer for directional biases optimizes a sum of sinusoids using 1 to 3 different frequencies, and keeping the best performing fit.

### Example

```{code-cell} ipython3
from xdem import biascorr

# Instantiate a directional bias correction
dirbias = biascorr.DirectionalBias(angle=65)
# Fit the data
dirbias.fit(ref_dem, tba_dem, inlier_mask=inlier_mask)

# Apply the transformation
corrected_dem = dirbias.apply(tba_dem)
```

## Terrain biases

{class}`xdem.biascorr.TerrainBias`

- **Performs:** Correct biases along a terrain attribute of the DEM.
- **Supports weights** Yes.
- **Recommended for:** Different native resolution between DEMs.

The default optimizer for terrain biases optimizes a 1D polynomial with an order from 1 to 6, and keeping the best performing fit.

### Example

```{code-cell} ipython3
from xdem import biascorr

# Instantiate a 1st order terrain bias correction
terbias = biascorr.TerrainBias(terrain_attribute="maximum_curvature")
# Fit the data
terbias.fit(ref_dem, tba_dem, inlier_mask=inlier_mask)

# Apply the transformation
corrected_dem = terbias.apply(tba_dem)
```

## Generic 1-D, 2-D and N-D classes

All bias-corrections methods are inherited from generic classes that perform corrections in 1-, 2- or N-D. Having these
separate helps the user navigating the dimensionality of the functions, optimizer, binning or variables used.

{class}`xdem.biascorr.BiasCorr1D`
{class}`xdem.biascorr.BiasCorr2D`
{class}`xdem.biascorr.BiasCorrND`

- **Performs:** Correct biases with any function and optimizer, or any binning, in 1-, 2- or N-D.
- **Supports weights** Yes.
- **Recommended for:** Anything.

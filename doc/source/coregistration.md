---
file_format: mystnb
mystnb:
  execution_timeout: 150
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
(coregistration)=

# Coregistration

xDEM implements a wide range of **coregistration algorithms and pipelines for 3-dimensional alignment** from the
peer-reviewed literature often tailored specifically to elevation data, aiming at correcting systematic elevation errors.

Two categories of alignment are generally differentiated: 3D [affine transformations](https://en.wikipedia.org/wiki/Affine_transformation) 
described below, and other alignments that possibly rely on external variables, described in {ref}`biascorr`.

Affine transformations can include vertical and horizontal translations, rotations and reflections, and scalings.

:::{admonition} More reading
:class: tip

Coregistration heavily relies on the use of static surfaces, which you can read more about on the **{ref}`static-surfaces` guide page**.

:::

## Quick use

Coregistration pipelines are defined by combining {class}`~xdem.coreg.Coreg` objects:

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9  # Default 10 is a bit too big for coregistration plots
```

```{code-cell} ipython3
import xdem

# Create a coregistration pipeline
my_coreg_pipeline = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
```

Then, coregistering a pair of elevation data can be done by calling {func}`xdem.DEM.coregister_3d` from the DEM that should be aligned.

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

```{code-cell} ipython3
# Coregister by calling the DEM method
aligned_dem = tba_dem.coregister_3d(ref_dem, my_coreg_pipeline)
```

Alternatively, the coregistration can be applied by calling {func}`~xdem.coreg.Coreg.fit_and_apply`, or sequentially 
calling the {func}`~xdem.coreg.Coreg.fit` and {func}`~xdem.coreg.Coreg.apply` steps,
which allows a broader variety of arguments at each step, and re-using the same transformation to several objects 
(e.g., horizontal shift of both a stereo DEM and its ortho-image).

```{code-cell} ipython3
# (Equivalent) Or use fit and apply
aligned_dem = my_coreg_pipeline.fit_and_apply(ref_dem, tba_dem)
```

Information about the coregistration inputs and outputs is summarized in {func}`~xdem.coreg.Coreg.info`.

```{tip}
Often, an `inlier_mask` has to be passed to {func}`~xdem.coreg.Coreg.fit` to isolate static surfaces to utilize during coregistration (for instance removing vegetation, snow, glaciers). This mask can be easily derived using {func}`~geoutils.Vector.create_mask`.
```

## Using a coregistration  

(coreg_object)=
### The {class}`~xdem.coreg.Coreg` object

Each coregistration method implemented in xDEM inherits their interface from the {class}`~xdem.coreg.Coreg` class<sup>1</sup>, and has the following methods:
- {func}`~xdem.coreg.Coreg.fit_and_apply` for estimating the transformation and applying it in one step,
- {func}`~xdem.coreg.Coreg.info` for plotting the metadata, including inputs and outputs of the coregistration.

The two above methods cover most uses. More specific methods are also available:
- {func}`~xdem.coreg.Coreg.fit` for estimating the transformation without applying it,
- {func}`~xdem.coreg.Coreg.apply` for applying an estimated transformation,
- {func}`~xdem.coreg.AffineCoreg.to_matrix` to convert the transform to a 4x4 transformation matrix, if possible,
- {func}`~xdem.coreg.AffineCoreg.from_matrix` to create a coregistration from a 4x4 transformation matrix.

```{margin}
<sup>1</sup>In a style inspired by [scikit-learn's pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn-linear-model-linearregression).
```

**Inheritance diagram of implemented coregistrations:**

```{eval-rst}
.. inheritance-diagram:: xdem.coreg.base.Coreg xdem.coreg.affine xdem.coreg.biascorr
        :top-classes: xdem.coreg.Coreg
```

See {ref}`biascorr` for more information on non-rigid transformations ("bias corrections").

### Accessing coregistration metadata

The metadata surrounding a coregistration method, which can be displayed by {func}`~xdem.coreg.Coreg.info`, is stored in 
the {attr}`~xdem.coreg.Coreg.meta` nested dictionary.
This metadata is divided into **inputs** and **outputs**. Input metadata corresponds to what arguments are 
used when initializing a {class}`~xdem.coreg.Coreg` object, while output metadata are created during the call to 
{func}`~xdem.coreg.Coreg.fit`. Together, they allow to apply the transformation during the
{func}`~xdem.coreg.Coreg.apply` step of the coregistration.

```{code-cell} ipython3
# Example of metadata info after fitting
my_coreg_pipeline.info()
```

For both **inputs** and **outputs**, four consistent categories of metadata are defined.

**Note:** Some metadata, such as the parameters `fit_or_bin` and `fit_func` described below, are pre-defined for affine coregistration methods and cannot be modified. They only take user-specified value for {ref}`biascorr`.

**1. Randomization metadata (common to all)**: 

- An input `subsample` to define the subsample size of valid data to use in all methods (recommended for performance), 
- An input `random_state` to define the random seed for reproducibility of the subsampling (and potentially other random aspects such as optimizers), 
- An output `subsample_final` that stores the final subsample size used, which can be smaller than requested depending on the amount of valid data intersecting the two elevation datasets. 

**2. Fitting and binning metadata (common to nearly all methods)**: 

- An input `fit_or_bin` to either fit a parametric model by passing **"fit"**, perform an empirical binning by passing **"bin"**, or to fit a parametric model to the binning with **"bin_and_fit" (only "fit" or "bin_and_fit" possible for affine methods)**,
- An input `fit_func` to pass any parametric function to fit to the bias **(pre-defined for affine methods)**,
- An input `fit_optimizer` to pass any optimizer function to perform the fit minimization,
- An input `bin_sizes` to pass the size or edges of the bins for each variable,
- An input `bin_statistic` to pass the statistic to compute in each bin,
- An input `bin_apply_method` to pass the method to apply the binning for correction,
- An output `fit_params` that stores the optimized parameters for `fit_func`,
- An output `fit_perr` that stores the error of optimized parameters (only for default `fit_optimizer`),
- An output `bin_dataframe` that stores the dataframe of binned statistics.

**3. Iteration metadata (common to all iterative methods)**: 

- An input `max_iterations` to define the maximum number of iterations at which to stop the method,
- An input `tolerance` to define the tolerance at which to stop iterations (tolerance unit defined in method description),
- An output `last_iteration` that stores the last iteration of the method,
- An output `all_tolerances` that stores the tolerances computed at each iteration.

**4. Affine metadata (common to all affine methods)**:

- An output `matrix` that stores the estimated affine matrix,
- An output `centroid` that stores the centroid coordinates with which to apply the affine transformation,
- Outputs `shift_x`, `shift_y` and `shift_z` that store the easting, northing and vertical offsets, respectively.

```{tip}
In xDEM, you can extract the translations and rotations of an affine matrix using {class}`xdem.coreg.AffineCoreg.to_translations` and 
{class}`xdem.coreg.AffineCoreg.to_rotations`.

To further manipulate affine matrices, see the [documentation of pytransform3d](https://dfki-ric.github.io/pytransform3d/rotations.html).
```

**5. Specific metadata (only for certain methods)**:

These metadata are only inputs specific to a given method, outlined in the method description.

For instance, for {class}`xdem.coreg.Deramp`, an input `poly_order` to define the polynomial order used for the fit, and 
for {class}`xdem.coreg.DirectionalBias`, an input `angle` to define the angle at which to do the directional correction.

## Coregistration methods

```{important}
Below we **create misaligned elevation data to examplify the different methods** in relation to their type of affine transformation.

See coregistration on real data in the **{ref}`examples-basic` and {ref}`examples-advanced` gallery examples**!
```

(coregistration-nuthkaab)=
### Nuth and Kääb (2011)

{class}`xdem.coreg.NuthKaab`

- **Performs:** Horizontal and vertical shifts.
- **Supports weights:** Planned.
- **Pros:** Refines sub-pixel horizontal shifts accurately, with fast convergence.
- **Cons:** Diverges on flat terrain, as landforms are required to constrain the fit with aspect and slope.

The [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011) coregistration approach estimates a horizontal
translation iteratively by solving a cosine equation between the terrain slope, aspect and the elevation differences.
The iteration stops if it reaches the maximum number of iteration limit or if the tolerance does not improve.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for adding a horizontal and vertical shift"
:  code_prompt_hide: "Hide the code for adding a horizontal and vertical shift"

x_shift = 30
y_shift = 30
z_shift = 10
# Affine matrix for 3D transformation
matrix = np.array(
    [
        [1, 0, 0, x_shift],
        [0, 1, 0, y_shift],
        [0, 0, 1, z_shift],
        [0, 0, 0, 1],
    ]
)
# We create misaligned elevation data
tba_dem_shifted = xdem.coreg.apply_matrix(ref_dem, matrix)
```

```{code-cell} ipython3
# Define a coregistration based on the Nuth and Kääb (2011) method
nuth_kaab = xdem.coreg.NuthKaab()
# Fit to data and apply
aligned_dem = nuth_kaab.fit_and_apply(ref_dem, tba_dem_shifted)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before NK")
(tba_dem_shifted - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After NK")
(aligned_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

### Vertical shift

{class}`xdem.coreg.VerticalShift`

- **Performs:** Vertical shifting using any custom function (mean, median, percentile).
- **Supports weights:** Planned.
- **Pros:** Useful to have as independent step to refine vertical alignment precisely as it is the most sensitive to outliers, by refining inliers and the central estimate function.
- **Cons**: Always needs to be combined with another approach.

The vertical shift coregistration is simply a shift based on an estimate of the mean elevation differences with customizable arguments.


```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for adding a vertical shift"
:  code_prompt_hide: "Hide the code for adding a vertical shift"

# Apply a vertical shift of 10 meters
tba_dem_vshifted = ref_dem + 10
```

```{code-cell} ipython3
# Define a coregistration object based on a vertical shift correction
vshift = xdem.coreg.VerticalShift(vshift_reduc_func=np.median)
# Fit and apply
aligned_dem = vshift.fit_and_apply(ref_dem, tba_dem_vshifted)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before vertical\nshift")
(tba_dem_vshifted - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After vertical\nshift")
(aligned_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

(icp)=

### Iterative closest point

{class}`xdem.coreg.ICP`

- **Performs:** Rigid transform transformation (3D translation + 3D rotation).
- **Does not support weights.**
- **Pros:** Efficient at estimating rotation and shifts simultaneously.
- **Cons:** Poor sub-pixel accuracy for horizontal shifts, sensitive to outliers, and runs slowly with large samples.

Iterative Closest Point (ICP) coregistration is an iterative point cloud registration method from [Besl and McKay (1992)](https://doi.org/10.1117/12.57955). It aims at iteratively minimizing the closest distance by apply sequential rigid transformations. If DEMs are used as inputs, they are converted to point clouds.
As for Nuth and Kääb (2011), the iteration stops if it reaches the maximum number of iteration limit or if the tolerance does not improve.

ICP is currently based on [OpenCV's implementation](https://docs.opencv.org/4.x/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html) (an optional dependency), which includes outlier removal arguments. This may improve results significantly on outlier-prone data, but care should still be taken, as the risk of landing in [local minima](https://en.wikipedia.org/wiki/Maxima_and_minima) increases.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for adding a shift and rotation"
:  code_prompt_hide: "Hide the code for adding a shift and rotation"

# Apply a rotation of 0.2 degrees and X/Y/Z shifts to elevation in meters
rotation = np.deg2rad(0.2)
x_shift = 100
y_shift = 200
z_shift = 50
# Affine matrix for 3D transformation
matrix = np.array(
    [
        [1, 0, 0, x_shift],
        [0, np.cos(rotation), -np.sin(rotation), y_shift],
        [0, np.sin(rotation), np.cos(rotation), z_shift],
        [0, 0, 0, 1],
    ]
)
centroid = [ref_dem.bounds.left + 5000, ref_dem.bounds.top - 2000, np.median(ref_dem) + 100]
# We create misaligned elevation data
tba_dem_shifted_rotated = xdem.coreg.apply_matrix(ref_dem, matrix, centroid=centroid)
```

```{code-cell} ipython3
# Define a coregistration based on ICP
icp = xdem.coreg.ICP()
# Fit to data and apply
aligned_dem = icp.fit_and_apply(ref_dem, tba_dem_shifted_rotated)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before ICP")
(tba_dem_shifted_rotated - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After ICP")
(aligned_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

## Building coregistration pipelines

### The {class}`~xdem.coreg.CoregPipeline` object

Often, more than one coregistration approach is necessary to obtain the best results, and several need to be combined 
sequentially. A {class}`~xdem.coreg.CoregPipeline` can be constructed for this:

```{code-cell} ipython3
# We can list sequential coregistration methods to apply
pipeline = xdem.coreg.CoregPipeline([xdem.coreg.ICP(), xdem.coreg.NuthKaab()])

# Or sum them, which works identically as the syntax above
pipeline = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
```

The {class}`~xdem.coreg.CoregPipeline` object exposes the same interface as the {class}`~xdem.coreg.Coreg` object.
The results of a pipeline can be used in other programs by exporting the combined transformation matrix using {func}`~xdem.coreg.Coreg.to_matrix`.

```{margin}
<sup>2</sup>Here again, this class is heavily inspired by SciKit-Learn's [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn-pipeline-pipeline) and [make_pipeline()](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline) functionalities.
```

```{code-cell} ipython3
# Fit to data and apply the pipeline of ICP + Nuth and Kääb
aligned_dem = pipeline.fit_and_apply(ref_dem, tba_dem_shifted_rotated)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before ICP + NK")
(tba_dem_shifted_rotated - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After ICP + NK")
(aligned_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

### Recommended pipelines

To ensure sub-pixel accuracy, the [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011) coregistration should almost always be used as a final step.
The approach does not account for rotations in the dataset, however, so a combination is often necessary.
For small rotations, a 1st degree deramp can be used in combination:

```{code-cell} ipython3
pipeline = xdem.coreg.NuthKaab() + xdem.coreg.Deramp(poly_order=1)
```

For larger rotations, ICP can be used instead:

```{code-cell} ipython3
pipeline = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
```

Additionally, ICP tends to fail with large initial vertical differences, so a preliminary vertical shifting can be used:

```{code-cell} ipython3
pipeline = xdem.coreg.VerticalShift() + xdem.coreg.ICP() + xdem.coreg.NuthKaab()
```

## Dividing coregistration in blocks

### The {class}`~xdem.coreg.BlockwiseCoreg` object

Sometimes, we want to split a coregistration across different spatial subsets of an elevation dataset, running that 
method independently in each subset. A {class}`~xdem.coreg.BlockwiseCoreg` can be constructed for this:

```{code-cell} ipython3
blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), subdivision=16)
```

The subdivision corresponds to an equal-length block division across the extent of the elevation dataset. It needs 
to be a number of the form 2{sup}`n` (such as 4 or 256).

It is run the same way as other coregistrations:

```{code-cell} ipython3
# Run 16 block coregistrations
aligned_dem = blockwise.fit_and_apply(ref_dem, tba_dem_shifted)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before block NK")
(tba_dem_shifted - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After block NK")
(aligned_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```
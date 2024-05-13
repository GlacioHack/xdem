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
(coregistration)=

# Coregistration

xDEM implements a wide range of **coregistration algorithms and pipelines for 3-dimensional alignment** from the
peer-reviewed literature often tailored specifically to elevation data, aiming at correcting systematic elevation errors.

Two categories of alignment are generally differentiated: **3D affine transformations** described below, and other
alignments that possibly rely on external variables described in {ref}`biascorr`.


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

Alternatively, the coregistration can be applied by sequentially calling the {func}`~xdem.coreg.Coreg.fit` and {func}`~xdem.coreg.Coreg.apply` steps,
which allows a broader variety of arguments at each step, and re-using the same transformation to several objects (e.g., horizontal shift of both a stereo DEM and its ortho-image).

```{code-cell} ipython3
# (Equivalent) Or, use fit and apply in two calls
my_coreg_pipeline.fit(ref_dem, tba_dem)
aligned_dem = my_coreg_pipeline.apply(tba_dem)
```

```{tip}
Often, an `inlier_mask` has to be passed to {func}`~xdem.coreg.Coreg.fit` to isolate static surfaces to utilize during coregistration (for instance removing vegetation, snow, glaciers). This mask can be easily derived using {func}`~geoutils.Vector.create_mask`.
```

## What is coregistration?

Coregistration is the process of finding a transformation to align data in a certain number of dimensions. In the case
of elevation data, in three dimensions.

Transformations that can be described by a 3-dimensional [affine](https://en.wikipedia.org/wiki/Affine_transformation)
function are included in coregistration methods, which include:

- vertical and horizontal translations,
- rotations, reflections,
- scalings.

(coreg_object)=
## The {class}`~xdem.coreg.Coreg` object

Each coregistration method implemented in xDEM inherits their interface from the {class}`~xdem.coreg.Coreg` class<sup>1</sup>, and has the following methods:
- {func}`~xdem.coreg.Coreg.fit` for estimating the transform.
- {func}`~xdem.coreg.Coreg.apply` for applying the transform to a DEM.
- {func}`~xdem.coreg.AffineCoreg.to_matrix` to convert the transform to a 4x4 transformation matrix, if possible.
- {func}`~xdem.coreg.AffineCoreg.from_matrix` to create a coregistration from a 4x4 transformation matrix.

```{margin}
<sup>1</sup>In a style inspired by [scikit-learn's pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn-linear-model-linearregression).
```

First, {func}`~xdem.coreg.Coreg.fit` is called to estimate the transform, and then this transform can be used or exported using the subsequent methods.

**Inheritance diagram of implemented coregistrations:**

```{eval-rst}
.. inheritance-diagram:: xdem.coreg.base.Coreg xdem.coreg.affine xdem.coreg.biascorr
        :top-classes: xdem.coreg.Coreg
```

See {ref}`biascorr` for more information on non-rigid transformations ("bias corrections").

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
tba_dem_shift = xdem.coreg.apply_matrix(ref_dem, matrix)
```

```{code-cell} ipython3
# Define a coregistration based on the Nuth and Kääb (2011) method
nuth_kaab = xdem.coreg.NuthKaab()
# Fit to data and apply
nuth_kaab.fit(ref_dem, tba_dem_shift)
aligned_dem = nuth_kaab.apply(tba_dem_shift)
```


```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before NK")
(tba_dem_shift - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After NK")
(aligned_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

### Tilt

{class}`xdem.coreg.Tilt`

- **Performs:** A 2D plane tilt correction.
- **Supports weights:** Planned.
- **Pros:** Corrects small rotations fairly accurately, and runs very fast.
- **Cons:** Not perfectly equivalent to a rotational correction, to use only with small rotations. For large rotational corrections, {ref}`icp` is recommended.

Tilt coregistration works by estimating and correcting for a 2D first-order polynomial (plane) over the entire elevation differences.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for adding a tilt"
:  code_prompt_hide: "Hide the code for adding a tilt"

# Apply a rotation of 0.2 degrees
rotation = np.deg2rad(0.2)
matrix = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(rotation), -np.sin(rotation), 0],
        [0, np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 0, 1],
    ]
)
# We create misaligned elevation data
tba_dem_tilt = xdem.coreg.apply_matrix(ref_dem, matrix)
```

```{code-cell} ipython3
# Define a coregistration based on a tilt correction
tilt = xdem.coreg.Tilt()
# Fit to data and apply
tilt.fit(ref_dem, tba_dem_tilt)
aligned_dem = tilt.apply(tba_dem_tilt)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before de-tilt")
(tba_dem_tilt - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After de-tilt")
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
tba_dem_vshift = ref_dem + 10
```

```{code-cell} ipython3
# Define a coregistration object based on a vertical shift correction
vshift = xdem.coreg.VerticalShift(vshift_func=np.median)
# Fit and apply
vshift.fit(ref_dem, tba_dem_vshift)
aligned_dem = vshift.apply(tba_dem_vshift)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before vertical\nshift")
(tba_dem_vshift - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0])
ax[1].set_title("After vertical\nshift")
(aligned_dem - ref_dem).plot(cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

(icp)=

### ICP

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
x_shift = 20
y_shift = 20
z_shift = 5
# Affine matrix for 3D transformation
matrix = np.array(
    [
        [1, 0, 0, x_shift],
        [0, np.cos(rotation), -np.sin(rotation), y_shift],
        [0, np.sin(rotation), np.cos(rotation), z_shift],
        [0, 0, 0, 1],
    ]
)
# We create misaligned elevation data
tba_dem_shifted_rotated = xdem.coreg.apply_matrix(ref_dem, matrix)
```

```{code-cell} ipython3
# Define a coregistration based on ICP
icp = xdem.coreg.ICP()
# Fit to data and apply
icp.fit(ref_dem, tba_dem_shifted_rotated)
aligned_dem = icp.apply(tba_dem_shifted_rotated)
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

## The {class}`~xdem.coreg.CoregPipeline` object

Often, more than one coregistration approach is necessary to obtain the best results. For example, ICP works poorly with large initial vertical shifts, so a {class}`~xdem.coreg.CoregPipeline` can be constructed to perform both sequentially:

```{code-cell} ipython3
# We can list sequential coregistration methods to apply
pipeline = xdem.coreg.CoregPipeline([xdem.coreg.ICP(), xdem.coreg.NuthKaab()])

# Or sum them, which works identically as the syntax above
pipeline = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
```

The {class}`~xdem.coreg.CoregPipeline` object exposes the same interface as the {class}`~xdem.coreg.Coreg` object.
The results of a pipeline can be used in other programs by exporting the combined transformation matrix using {func}`~xdem.coreg.CoregPipeline.to_matrix`.

```{margin}
<sup>2</sup>Here again, this class is heavily inspired by SciKit-Learn's [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn-pipeline-pipeline) and [make_pipeline()](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline) functionalities.
```

```{code-cell} ipython3
# Fit to data and apply the pipeline of ICP + Nuth and Kääb
pipeline.fit(ref_dem, tba_dem_shifted_rotated)
aligned_dem = pipeline.apply(tba_dem_shifted_rotated)
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
pipeline = xdem.coreg.Tilt() + xdem.coreg.NuthKaab()
```

For larger rotations, ICP can be used instead:

```{code-cell} ipython3
pipeline = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
```

Additionally, ICP tends to fail with large initial vertical differences, so a preliminary vertical shifting can be used:

```{code-cell} ipython3
pipeline = xdem.coreg.VerticalShift() + xdem.coreg.ICP() + xdem.coreg.NuthKaab()
```

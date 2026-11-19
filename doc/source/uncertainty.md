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
(uncertainty)=

# Uncertainty

xDEM integrates tools from the literature that rely on **joint methods** from two
scientific fields: **spatial statistics** and **uncertainty quantification**.

While uncertainty analysis technically refers to both systematic and random errors, systematic errors of elevation data
are corrected using {ref}`coregistration` and {ref}`biascorr`, so we here refer to uncertainty analysis for **quantifying and
propagating random errors** (including structured errors).

In detail, xDEM provides tools to:

1. Estimate and model **spatially variable** errors (e.g., depending on terrain slope),
2. Estimate and model **spatially correlated** errors (e.g., large-scale noise),
3. Propagate uncertainties to **any operation** (e.g., a spatial average, coregistration, or slope).

```{important}
xDEM's uncertainty API has been **refactored**, for ease-of-use and scalability!

It now relies on a [generic spatial uncertainty module in GeoUtils](https://geoutils.readthedocs.io/en/stable/uncertainty.html)
on which xDEM builds upon to add tools for **terrain** and **3D coregistration**, supporting **datasets larger than memory**.

See {ref}`uncertainty-module-migration` for adjusting from the previous API.
```

:::{admonition} More reading
:class: tip

For an introduction, see the guides pages on **{ref}`spatial-stats`** and **{ref}`static-surfaces`**.
For variography, see [GeoUtils](https://geoutils.readthedocs.io/en/stable/stats.html#stats-spatial) or [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/)'s pages.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# Match the figure resolution and text size used by the other feature pages
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9
```

(uncertainty-quick)=
## Quick use

We rely on **differences on static surfaces** as an **error proxy** to estimate error structure, which
can in turn be used to propagate uncertainty.

Below, we compare two DEMs near Longyearbyen, excluding glacierized terrain.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files and selecting stable terrain"
:  code_prompt_hide: "Hide the code for opening example files and selecting stable terrain"

import geoutils as gu
import numpy as np
import xdem

# Open the reference and moving DEMs
ref = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
tba = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines")).crop(tba)
static_surfaces = ~glacier_outlines.create_mask(tba)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the DEMs and their difference"
:  code_prompt_hide: "Hide the code for plotting the DEMs and their difference"

from matplotlib.patches import Patch

# Align a DEM for the difference map and glacier averages
dem = xdem.coreg.LZD().fit_and_apply(ref, tba, inlier_mask=static_surfaces, random_state=42)
dh = ref - dem
stable_dh = dh.copy()
stable_dh.set_mask(~static_surfaces)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
ref.plot(ax=axes[0], cmap="terrain", vmin=0, vmax=1000, cbar_title="Elevation (m)", title="Reference DEM")
tba.plot(ax=axes[1], cmap="terrain", vmin=0, vmax=1000, cbar_title="Elevation (m)", title="Other DEM")
stable_dh.plot(
    ax=axes[2], cmap="RdBu", vmin=-10, vmax=10,
    cbar_title="Elevation difference (m)", title="Error proxy",
)
glacier_outlines.plot(ref=stable_dh, ax=axes[2], fc="none", ec="black", linewidth=0.7)
axes[2].legend(
    handles=[Patch(facecolor="none", edgecolor="black", label="Excluded glacier surfaces")],
    loc="lower left", fontsize="small",
)
axes[2].set(xlabel="", ylabel="")
plt.show()
```

For estimation, we assume the **same** magnitude/correlation of errors for both DEMs, to avoid double-counting them.

We then define **two error components** to be estimated: one **short range** correlation with **variable** error, and one **long range** with **constant** error,
with Gaussian and stable models, respectively.

The estimation workflow then automatically uses [grouped statistics](https://geoutils.readthedocs.io/en/stable/stats.html#stats-grouped) and [variography](https://geoutils.readthedocs.io/en/stable/stats.html#stats-spatial) to constrain
these components. We select the **NMAD** and **Dowd** median-like estimators for the spread and empirical variogram respectively, to be robust to outliers.

```{code-cell} ipython3
# Estimate errors from elevation differences on static surfaces
es = tba.estimate_error_structure(
    ref, mask=static_surfaces, other_precision="same",
    spread_estimator=gu.stats.nmad, variogram_estimator="dowd", random_state=42,
    components={
        "short_range": {"magnitude": "heteroscedastic", "correlation": "gaussian"},
        "long_range": {"magnitude": "constant", "correlation": "stable"},
    },
)
```

:::{note}
By default, xDEM pre-coregisters the surfaces before estimating error structure, and uses **slope** and **maximal curvature**
as predictors of variability.
:::

The returned {class}`~xdem.ErrorStructure` describes **error magnitude and correlation** for either DEM:

```{code-cell} ipython3
es.plot(magnitude_kwargs={
    "xlabel": "Slope (°)",
    "ylabel": "Maximal curvature (100 / m)",
    "cbar_title": "Elevation\nerror (1σ, m)",
    "cmap": "Purples",
})
```

We can predict the **error magnitude** over the entire DEM, assuming errors inferred from static surfaces apply
to other terrain, such as glaciers:

```{code-cell} ipython3
# Derive the predictors needed to evaluate local error magnitudes
predictors = {"slope": tba.slope(), "max_curvature": tba.max_curvature()}
sig_dem = es.predict_magnitude(predictors, like=tba)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the predicted error map"
:  code_prompt_hide: "Hide the code for plotting the predicted error map"
:  image:
:    alt: "Predicted elevation error magnitude with glacier outlines."

fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
sig_dem.plot(ax=ax, cmap="Purples", vmin=0, cbar_title="Elevation error (1σ, m)", title="Predicted error")
glacier_outlines.plot(ref=sig_dem, ax=ax, fc="none", ec="black", linewidth=0.7)
ax.set(xlabel="", ylabel="")
plt.show()
```

We can now propagate errors to **any operation**, such as the mean elevation difference over each glacier:

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for selecting the glacier areas"
:  code_prompt_hide: "Hide the code for selecting the glacier areas"

# Select both glaciers, combining outlines with the same name
glacier_names = ["Brombreen", "Medalsbreen"]
glacier_areas = glacier_outlines.gdf[glacier_outlines.gdf["NAME"].isin(glacier_names)].dissolve(by="NAME")
areas = [gu.Vector(glacier_areas.loc[[name]]) for name in glacier_names]
mean_differences = [np.nanmean(dh[area.create_mask(dh)]) for area in areas]
```

```{code-cell} ipython3
# Propagate errors to each glacier mean
glacier_uncertainties = xdem.uncertainty.spatial_error_propagation(
    areas, es, support=dh, predictors=predictors, random_state=42,
)
# Combine independent errors from both DEMs
glacier_uncertainties = np.sqrt(2) * np.asarray(glacier_uncertainties)
for name, mean, uncertainty in zip(glacier_names, mean_differences, glacier_uncertainties):
    print(f"{name}: {mean:.2f} ± {uncertainty:.2f} m (1σ)")
```

These uncertainties account for **variable error magnitudes and spatial correlation**:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the glacier uncertainties"
:  code_prompt_hide: "Hide the code for plotting the glacier uncertainties"
:  image:
:    alt: "Elevation differences around Brombreen and Medalsbreen."

glacier_dh = dh.crop(gu.Vector(glacier_areas).buffer(1800))
fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
glacier_dh.plot(ax=ax, cmap="RdYlBu", vmin=-30, vmax=30, cbar_title="Elevation difference (m)")

# Label each glacier mean and uncertainty
for name, area, mean, uncertainty in zip(glacier_names, areas, mean_differences, glacier_uncertainties):
    area.plot(ref=glacier_dh, ax=ax, fc="none", ec="black", linewidth=1)
    label = area.gdf.geometry.representative_point().iloc[0]
    ax.text(
        label.x, label.y, f"{name}\n{mean:.2f} ± {uncertainty:.2f} m",
        ha="center", va="center", fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
    )
ax.set(title="Mean elevation difference ± 1σ", xlabel="", ylabel="")
plt.show()
```

## Summary of methods

The methods of this module combine **spatial statistics** to model elevation errors and **uncertainty quantification** to
propagate their effect on estimated quantities. We distinguish the **error model** from the **calculation used
to propagate it**.

### Modelling error structure

Multiple studies address uncertainty analysis of elevation, or elevation changes, notably [Rolstad et al. (2009)](https://doi.org/10.3189/002214309789470950),
[Anderson (2019)](https://doi.org/10.1002/esp.4551) and [Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922).

For estimation, they are **mutually compatible**. The framework of [Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922) encompasses
the others, as it allows both **spatially variable error magnitudes** (heteroscedasticity) and **spatial correlation models with nested ranges**,
assuming a sum of components. This is directly supported by {class}`~xdem.ErrorStructure`, described further below in {ref}`uncertainty-object`.

### Propagating errors

Additionally, xDEM provides both **analytical solutions** for propagating uncertainty in some operations, and **numerical simulations**
to repeat the same operation on elevations perturbed by random error fields, allowing us to check whether the analytical
approximations are appropriate.

| Feature | Analytical propagation                                                   | References                                                                                                                                 |
| --- |--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| {ref}`Spatial averages <uncertainty-spatial-averages>` | Covariance sum, with circular or sampling approximations for large areas | [Rolstad et al. (2009)](https://doi.org/10.3189/002214309789470950); [Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922) |
| {ref}`Terrain attributes <uncertainty-terrain>` | First-order covariance propagation (Taylor expansion) for slope and aspect | [Oksanen & Sarjakoski (2005)](https://doi.org/10.1016/j.cageo.2005.02.014)                                                                 |
| {ref}`Coregistration parameters <uncertainty-coreg>` | Linearized parameter covariance for LZD and point-to-plane ICP           | [Bonnabel et al. (2016)](https://doi.org/10.1109/ACC.2016.7526532) for ICP; Hugonnet et al. (2027) for LZD                                 |

The {ref}`propagation examples <uncertainty-spatial-averages>` further below compare analytical and numerical propagation.

(uncertainty-object)=
## The error structure object

:::{seealso}
The [GeoUtils uncertainty page](https://geoutils.readthedocs.io/en/stable/uncertainty.html) contains
**additional examples and details** on spatial error propagation.
The sections below re-explain the core of these functionalities, adding **specifics for elevation data**,
so you can follow without having to refer to GeoUtils.
:::

Uncertainty features are centered around {class}`~xdem.ErrorStructure`, a **lightweight** object describing
error structure without storing a full covariance matrix or the variability predictors associated with it.

An {class}`~xdem.ErrorStructure` is composed of a sum of independent **components** to describe errors from
different sources, such as separating local random noise from large-scale error patterns.
Each component has:

- An error **magnitude** model, describing its statistical spread. It can be constant or vary with predictors such as
  terrain slope or curvature.
- An error **correlation** model, describing how errors relate across locations. Components can have
  different correlation ranges, or be spatially uncorrelated.

The figure below illustrates how an error structure **decomposes the total error into its components**,
using realizations of random fields multiplied by their predicted magnitudes (we reuse the error structure
from the {ref}`uncertainty-quick`):

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the plotting code on error components"
:  code_prompt_hide: "Hide the plotting code on error components"
:  image:
:    width: 1000px
:    alt: "A short range random field multiplied by terrain-dependent elevation error magnitude, plus an independent long range field multiplied by constant magnitude, equals the combined DEM error field."

# Use a smaller grid to illustrate the fitted components over the DEM extent
component_grid = tba.reproject(grid_size=(96, 96))
component_predictors = {name: value.reproject(component_grid) for name, value in predictors.items()}
short_magnitude = es.predict_magnitude(
    component_predictors, component="short_range", like=component_grid,
).to_nanarray().squeeze()
long_magnitude = es.predict_magnitude(component="long_range")

# Draw independent unit-variance fields with each component's fitted correlation
component_fields = []
for seed, name in enumerate(["short_range", "long_range"], start=42):
    unit_structure = xdem.ErrorStructure([xdem.ErrorComponent(name, 1.0, es.components[name].correlation)])
    field = component_grid.random_field(unit_structure, backend="gstools", random_state=seed)
    component_fields.append(field.to_nanarray().squeeze())
short_field, long_field = component_fields
combined_errors = short_magnitude * short_field + long_magnitude * long_field
magnitude_limit = max(np.nanmax(short_magnitude), long_magnitude)
component_limit = np.nanmax(np.abs(combined_errors))

# Multiply each pattern by its magnitude, then add the component errors
figure, axes = plt.subplots(
    1, 9, figsize=(13.5, 3.2), layout="constrained",
    gridspec_kw={"width_ratios": [1, 0.12, 1, 0.12, 1, 0.12, 1, 0.12, 1]},
)
panels = [
    (
        short_field, f"Short correlation range\n{es.components['short_range'].correlation.effective_range:.0f} m",
        "RdBu", -3, 3, "Standardized error",
    ),
    (
        short_magnitude, "Variable magnitude\nfrom slope/curvature", "Purples",
        0, magnitude_limit, "Error magnitude (1σ, m)",
    ),
    (
        long_field, f"Long correlation range\n{es.components['long_range'].correlation.effective_range:.0f} m",
        "RdBu", -3, 3, "Standardized error",
    ),
    (
        np.full_like(long_field, long_magnitude), "Constant magnitude", "Purples",
        0, magnitude_limit, "Error magnitude (1σ, m)",
    ),
    (combined_errors, "Combined errors", "RdBu", -component_limit, component_limit, "Elevation error (m)"),
]
for ax, (values, title, cmap, minimum, maximum, label) in zip(axes[::2], panels):
    image = ax.imshow(values, cmap=cmap, vmin=minimum, vmax=maximum, origin="upper")
    ax.set(title=title, xticks=[], yticks=[])
    figure.colorbar(image, ax=ax, orientation="horizontal", shrink=0.9, pad=0.15, label=label)

# Add the signs between maps
for ax, symbol in zip(axes[1::2], ["×", "+", "×", "="]):
    ax.set_axis_off()
    ax.text(0.5, 0.5, symbol, ha="center", va="center", fontsize=22, transform=ax.transAxes)
plt.show()
```

Structurally, an {class}`~xdem.ErrorStructure` contains one or more {class}`~xdem.ErrorComponent` objects,
each combining an {class}`~xdem.ErrorMagnitude` with an optional {class}`~geoutils.Variogram` correlation model.
For basic use, you should not have to manipulate those sub-objects, as estimation constructs them automatically. See
[GeoUtils' error model guide](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-model)
for more advanced, manual construction.

The components can be summarized with {meth}`~xdem.ErrorStructure.info`, or plotted with
{meth}`~xdem.ErrorStructure.plot` for all diagnostics, and {meth}`~xdem.ErrorStructure.plot_magnitude`
or {meth}`~xdem.ErrorStructure.plot_correlation` for each aspect:

```{code-cell} ipython3
es.info()
```

An error structure is returned by {meth}`~xdem.DEM.estimate_error_structure` and passed to the
{ref}`propagation functions <uncertainty-propagation>` or {ref}`random fields generation <uncertainty-simulation>`,
all described directly below.

(error-struc)=
## Estimate error structure

{meth}`DEM.estimate_error_structure() <xdem.DEM.estimate_error_structure>`<br>
{meth}`EPC.estimate_error_structure() <xdem.EPC.estimate_error_structure>`

Estimation combines **multiple statistical steps** to constrain an error structure, which can be inspected and reused for propagation.

```{seealso}
For advanced component definitions, refer to [GeoUtils' error estimation](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-estimation).
```

(spatialstats-heterosc)=
### Error proxy

Firstly, one needs a good **error proxy** to estimate errors.

For elevation data, differences on **static surfaces** (e.g. bare-rock, grasslands) are generally available.
From there, use `other_precision="same"` to assume to two DEMs have independent errors with the same magnitude and spatial correlation, or
`other_precision="negligible"` when reference errors can be ignored.

Below, we first **pre-coregister** the DEMs to correct systematic errors, and plot the final error proxy:

```{code-cell} ipython3
aligned_dem = xdem.coreg.LZD().fit_and_apply(ref, tba, inlier_mask=static_surfaces, random_state=42)
proxy = ref - aligned_dem
proxy.set_mask(~static_surfaces)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the error proxy"
:  code_prompt_hide: "Hide the code for plotting the error proxy"
:  image:
:    alt: "Elevation differences on static surfaces after coregistration, with glaciers excluded."

proxy.plot(cmap="RdBu", vmin=-10, vmax=10, cbar_title="Elevation difference (m)", title="Error proxy after coregistration")
plt.show()
```

Note that this pre-coregistration step can also be done later when calling {meth}`~xdem.DEM.estimate_error_structure`
by using the `precoreg` argument.

### Magnitude

Elevation errors often have **variable magnitude**, also called **heteroscedasticity**.

By default, xDEM uses terrain slope and maximal curvature to explain this variability.
Here, we pass only **slope** and explicitly select the **standard deviation** (while the default spread estimator is the **NMAD**,
for robustness to outliers):

```{code-cell} ipython3
slope_errors = tba.estimate_error_structure(
    ref, mask=static_surfaces, other_precision="same",
    predictors=("slope",), spread_estimator=np.std, bins=8, random_state=42,
    components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
)
slope_magnitude = slope_errors.predict_magnitude({"slope": tba.slope()}, like=tba)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the slope bins and predicted error magnitude"
:  code_prompt_hide: "Hide the code for plotting the slope bins and predicted error magnitude"
:  image:
:    width: 800px
:    alt: "Standard deviation of elevation errors in slope bins and the predicted error magnitude map."

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(left=0.08, right=0.95, bottom=0.2, top=0.8, wspace=0.5)
slope_errors.plot_magnitude(ax=axes[0], xlabel="Slope (°)", ylabel="Elevation error (1σ, m)", title="Slope bins")
slope_magnitude.plot(ax=axes[1], cmap="Purples", vmin=0, cbar_title="Elevation error (1σ, m)", title="Predicted error")
plt.show()
```

By default, **linear interpolation between bins** is used as functional model, but it is possible to specify any custom model,
and to refit without performing the binning again with {meth}`~xdem.ErrorStructure.refit_magnitude`, as shown in {ref}`uncertainty-custom-magnitude`.

We can pass several predictors, such as **slope and maximal curvature**:

```{code-cell} ipython3
terrain_errors = tba.estimate_error_structure(
    ref, mask=static_surfaces, other_precision="same",
    predictors=("slope", "max_curvature"), spread_estimator=gu.stats.nmad, bins=8, random_state=42,
    components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
)
```

Then use {meth}`~xdem.ErrorStructure.predict_magnitude` to predict an error magnitude map:

```{code-cell} ipython3
terrain_predictors = {"slope": tba.slope(), "max_curvature": tba.max_curvature()}
sig_dem = terrain_errors.predict_magnitude(terrain_predictors, like=tba)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the terrain bins and predicted error magnitude"
:  code_prompt_hide: "Hide the code for plotting the terrain bins and predicted error magnitude"
:  image:
:    width: 800px
:    alt: "NMAD of elevation errors in slope and curvature bins and the predicted error magnitude map."

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(left=0.08, right=0.95, bottom=0.2, top=0.8, wspace=0.5)
terrain_errors.plot_magnitude(
    ax=axes[0], xlabel="Slope (°)", ylabel="Maximal curvature (100 / m)",
    cbar_title="Elevation\nerror (1σ, m)", cmap="Purples", title="Slope and curvature bins",
)
sig_dem.plot(ax=axes[1], cmap="Purples", vmin=0, cbar_title="Elevation error (1σ, m)", title="Predicted error")
plt.show()
```

We can see that error magnitude varies both with slope and maximal curvature, despite some co-dependency, meaning
they are both useful to consider.

The map extends the fitted relationship beyond static surfaces, assuming that **the predictors also explain
errors on other terrain**, including glaciers.

For custom bins and magnitude models, see
[GeoUtils' magnitude examples](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-variable-magnitude).

### Correlation

**Spatial correlation** describes how elevation errors vary together across locations.

Below, we estimate a variogram, explicitly selecting the **Matheron** estimator (STD equivalent for semivariance/correlation,
while the default is **Dowd**, the robust median-equivalent for semivariance) and a **spherical** model fit:

```{code-cell} ipython3
single_range_errors = tba.estimate_error_structure(
    ref, mask=static_surfaces, other_precision="same",
    spread_estimator=gu.stats.nmad, variogram_estimator="matheron", random_state=42,
    components={"correlated": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
)
single_range_errors.plot_correlation(kind="correlation")
plt.show()
```

We see that the estimation is not great! So we instead use **Dowd**, and fit **two models with different ranges, a gaussian and a stable model**:

```{code-cell} ipython3
multi_range_errors = tba.estimate_error_structure(
    ref, mask=static_surfaces, other_precision="same",
    spread_estimator=gu.stats.nmad, variogram_estimator="dowd", random_state=42,
    components={
        "short_range": {"magnitude": "heteroscedastic", "correlation": "gaussian"},
        "long_range": {"magnitude": "constant", "correlation": "stable"},
    },
)
multi_range_errors.plot_correlation(kind="correlation")
plt.show()
```

To change the model forms while keeping the **same number of components**, use
{meth}`~xdem.ErrorStructure.refit_variogram`, as shown in the
{ref}`custom variogram example <uncertainty-custom-variogram>`.

For other estimators and models, see [GeoUtils' correlation examples](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-correlation).

(uncertainty-propagation)=
## Propagate uncertainty

{func}`xdem.uncertainty.propagate`

To propagate elevation errors **analytically** or **numerically**, pass an {class}`~xdem.ErrorStructure` to {func}`~xdem.uncertainty.propagate` along with the parameters of an operation,
which will return a {class}`~geoutils.PropagationSummary`.

Below, we first describe the {class}`~geoutils.PropagationSummary` object and how to generate random fields, then compare propagation methods for spatial averages, terrain attributes and coregistration fit.

(uncertainty-propagation-summary)=
### Propagation summary

Uncertainty propagation always returns a {class}`~geoutils.PropagationSummary` with:

- **{attr}`~geoutils.PropagationSummary.estimate`**: the result from the original elevations.
- **{attr}`~geoutils.PropagationSummary.mean`**: the propagated mean.
- **{attr}`~geoutils.PropagationSummary.std`**: the propagated 1σ uncertainty.

```{note}
For nonlinear operations, {attr}`~geoutils.PropagationSummary.mean` can differ from {attr}`~geoutils.PropagationSummary.estimate`.
```

Each one of these outputs **has the same type/shape as the typical output of the operation**. For example, a propagated uncertainty for a spatial average over 2 glaciers will return 2 scalar outputs,
a slope uncertainty will have the same shape as the slope raster, and a propagated coregistration uncertainty will return uncertainty for the 6 translation/rotations parameters.


```{important}
A summary **does not provide an {class}`~xdem.ErrorStructure`** as output. To propagate uncertainty through several
operations, bundle them into the same calculation.
```

For example, propagate errors to the **mean elevation of an 8×8 window**:

```{code-cell} ipython3
summary_dem = dem.icrop((100, 100, 108, 108))
summary_predictors = {name: values.crop(summary_dem) for name, values in predictors.items()}
mean_summary = xdem.uncertainty.propagate(
    summary_dem, "spatial", error_structure=es, predictors=summary_predictors,
    method="analytical", inputs={"spatial": {"areas": {"window": None}}}, at="all",
)
mean_summary.info()
```

Use {meth}`~geoutils.PropagationSummary.interval` to derive **95% uncertainty bounds**:

```{code-cell} ipython3
print(mean_summary.interval(0.95))
```

(uncertainty-simulation)=
### Random field generation

**Random error fields** are realizations of an {class}`~xdem.ErrorStructure`, generated with
{meth}`~xdem.DEM.random_field`.
Here, we use a 100 m grid to draw elevation errors from the fitted structure:

```{code-cell} ipython3
# Exclude cells without terrain predictors
field_dem = tba.reproject(res=100)
field_predictors = {name: values.reproject(field_dem).to_nanarray() for name, values in predictors.items()}
finite_predictors = np.logical_and.reduce([np.isfinite(values) for values in field_predictors.values()])
field_dem.set_mask(~finite_predictors)

# Fill masked predictors for generation over the full grid
field_predictors = {
    name: np.where(finite_predictors, values, np.median(values[finite_predictors]))
    for name, values in field_predictors.items()
}
field = field_dem.random_field(
    es, predictors=field_predictors, backend="gstools", random_state=42,
)
```

```{note}
Random field generation **relies on [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) or [GPyTorch](https://docs.gpytorch.ai/en/stable/)**. The former supports more correlation forms, while the latter is usually faster.

GeoUtils adds logic to **directly support complex error structure** from multiple-range correlation models and variable magnitude from predictors, applicable on large datasets.
```

The examples below emphasize variable magnitudes and increasingly long correlation ranges:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for comparing random fields"
:  code_prompt_hide: "Hide the code for comparing random fields"
:  image:
:    width: 1000px
:    alt: "Five elevation error fields with constant or variable magnitude, uncorrelated errors, short range correlation and several correlation ranges."

# Match average variance across the comparison models
variable_magnitude = terrain_errors.components["uncorrelated"].magnitude
field_magnitudes = terrain_errors.predict_magnitude(field_predictors, like=field_dem)
field_magnitude = float(np.sqrt(np.nanmean(field_magnitudes.to_nanarray() ** 2)))

# Compare uncorrelated errors with a 1 km correlation range
field_short_model = gu.Variogram.from_model("gaussian", effective_range=1000, partial_sill=1)
uncorrelated_model = xdem.ErrorStructure([xdem.ErrorComponent("uncorrelated", field_magnitude)])
variable_model = xdem.ErrorStructure([xdem.ErrorComponent("uncorrelated", variable_magnitude)])
correlated_model = xdem.ErrorStructure([xdem.ErrorComponent("correlated", field_magnitude, field_short_model)])
variable_correlated_model = xdem.ErrorStructure([
    xdem.ErrorComponent("correlated", variable_magnitude, field_short_model),
])

# Combine patterns spanning 200 m, 1 km and 4 km
range_models = [
    gu.Variogram.from_model("gaussian", effective_range=distance, partial_sill=variance)
    for distance, variance in [(200, 0.1), (1000, 0.3), (4000, 0.6)]
]
mixed_correlation = gu.Variogram.combine(*range_models)
mixed_model = xdem.ErrorStructure([xdem.ErrorComponent("correlated", variable_magnitude, mixed_correlation)])

# Draw each model over the same DEM grid
uncorrelated_field = field_dem.random_field(uncorrelated_model, random_state=42)
variable_uncorrelated_field = field_dem.random_field(variable_model, predictors=field_predictors, random_state=42)
correlated_field = field_dem.random_field(correlated_model, backend="gstools", random_state=42)
variable_correlated_field = field_dem.random_field(
    variable_correlated_model, predictors=field_predictors, backend="gstools", random_state=42,
)
mixed_field = field_dem.random_field(
    mixed_model, predictors=field_predictors, backend="gstools", random_state=42,
)

# Use one color scale to compare the error patterns
comparison_fields = [
    uncorrelated_field, variable_uncorrelated_field, correlated_field, variable_correlated_field, mixed_field,
]
field_limit = 3 * field_magnitude
figure, axes = plt.subplots(1, 5, figsize=(12, 3.5), sharex=True, sharey=True, layout="compressed")
figure.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.02)
for ax, values, title in zip(
    axes, comparison_fields,
    [
        "Uncorrelated\nConstant magnitude", "Uncorrelated\nVariable magnitude",
        "Short range\nConstant magnitude", "Short range\nVariable magnitude",
        "Several ranges\nVariable magnitude",
    ],
):
    values.plot(ax=ax, cmap="RdBu", vmin=-field_limit, vmax=field_limit, add_cbar=False)
    ax.set(title=title, xlabel="", ylabel="", xticks=[], yticks=[])
figure.colorbar(
    axes[0].images[0], ax=axes, orientation="horizontal", shrink=0.7, aspect=40,
    fraction=0.06, pad=0.08, extend="both", label="Elevation error (m)",
)
plt.show()
```

The field contains **errors only**. Add it to the DEM to create a perturbed input for a simulation:

```{code-cell} ipython3
dem_simulated = field_dem + field
```

(uncertainty-spatial-averages)=
### Spatial averages

Correlated errors **average out more slowly** than independent errors.

Below, we propagate errors to the mean elevation differences over **Brombreen and Medalsbreen glaciers**, as in {ref}`uncertainty-quick`.

**Analytical** propagation uses generic covariance propagation, as described on the **{ref}`spatialstats-errorpropag` guide page**,
with a choice of approximations from [Rolstad et al. (2009)](https://doi.org/10.3189/002214309789470950)
or [Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922).

**Numerical** propagation generates multiple {ref}`random error fields <uncertainty-simulation>` to perturb the input, and compute the output multiple times.

We combine independent errors from both DEMs and use a 100 m grid to speed up numerical simulations:

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for preparing the glacier comparison"
:  code_prompt_hide: "Hide the code for preparing the glacier comparison"

# Combine independent errors from the reference and moving DEMs
change_errors = xdem.ErrorStructure([
    xdem.ErrorComponent(f"{source}_{name}", component.magnitude, component.correlation)
    for source in ["ref", "tba"]
    for name, component in es.components.items()
])
mean_grid = glacier_dh.reproject(res=100)
mean_predictors = {name: values.reproject(mean_grid) for name, values in predictors.items()}
mean_inputs = {"spatial": {"areas": dict(zip(glacier_names, areas))}}
```

```{code-cell} ipython3
analytical_means = xdem.uncertainty.propagate(
    mean_grid, "spatial", error_structure=change_errors, predictors=mean_predictors,
    inputs=mean_inputs, method="analytical", at="all",
)
numerical_means = xdem.uncertainty.propagate(
    mean_grid, "spatial", error_structure=change_errors, predictors=mean_predictors,
    inputs=mean_inputs, method="numerical", at="all", return_samples=True,
    n_simulations=100, random_state=42, random_field_options={"backend": "gstools"},
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for comparing glacier mean uncertainties"
:  code_prompt_hide: "Hide the code for comparing glacier mean uncertainties"

import pandas as pd

plt.rcParams["font.size"] = 11
mean_comparison = pd.DataFrame({
    "Analytical": analytical_means.std, "Numerical (100 simulations)": numerical_means.std,
})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")

# Compare the simulated glacier means
numerical_means.samples[glacier_names].plot.hist(
    ax=axes[0], bins=15, color=["C0", "C1"], alpha=0.5,
    xlabel="Mean elevation difference (m)", ylabel="Number of simulations", title="Numerical glacier means",
)
axes[0].get_legend().set_title("")

# Mark mean ±1σ at a fixed height
for name in glacier_names:
    mean = numerical_means.mean[name]
    sigma = numerical_means.std[name]
    axes[0].errorbar(
        mean, 0.55, xerr=sigma, fmt="o", color="black", capsize=4,
        transform=axes[0].get_xaxis_transform(), zorder=3,
    )
    axes[0].text(
        mean, 0.62, f"±1σ = ±{sigma:.2f} m", ha="center",
        transform=axes[0].get_xaxis_transform(),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )

# Compare analytical and numerical uncertainties
mean_comparison.plot.bar(
    ax=axes[1], color=["#756bb1", "#bcbddc"], rot=0,
    ylim=(0, 1.25 * mean_comparison.to_numpy().max()),
    xlabel="", ylabel="Mean elevation difference error (1σ, m)", title="Glacier mean uncertainty",
)
plt.show()
```

We see a good agreement between analytical and numerical, with a slight overestimation through analytical!

One should increase the number of simulations to check convergence. For **random field** backends and approximations, see
[GeoUtils' simulation guide](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-simulation).

(uncertainty-terrain)=
### Terrain attributes

We can also propagate errors to terrain attributes such as **slope and aspect**.

**Analytical** propagation uses a **first-order Taylor series approximation**, following
[Oksanen & Sarjakoski (2005)](https://doi.org/10.1016/j.cageo.2005.02.014).

**Numerical** propagation generates multiple {ref}`random error fields <uncertainty-simulation>` to perturb the input, and compute the output multiple times, as for other operations.

Below, we compare slope uncertainty from both methods with the "Florinsky" surface fit:

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for selecting a small area"
:  code_prompt_hide: "Hide the code for selecting a small area"

# Select a 600 × 600 m area on the original 20 m grid
detail_dem = dem.icrop((500, 450, 530, 480))
detail_predictors = {name: values.crop(detail_dem) for name, values in predictors.items()}
terrain_options = {"attribute": "slope", "surface_fit": "Florinsky"}
```

```{code-cell} ipython3
analytical_slope = xdem.uncertainty.propagate(
    detail_dem, "terrain", error_structure=es, predictors=detail_predictors,
    method="analytical", operation_options=terrain_options,
)
numerical_slope = xdem.uncertainty.propagate(
    detail_dem, "terrain", error_structure=es, predictors=detail_predictors,
    method="numerical", operation_options=terrain_options,
    n_simulations=200, random_state=42, random_field_options={"backend": "gstools"},
    at=[(15, 15)], return_samples=True,
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for comparing slope uncertainties"
:  code_prompt_hide: "Hide the code for comparing slope uncertainties"
:  image:
:    width: 1000px
:    alt: "Slope and the simulated slope distribution above analytical and numerical slope uncertainty maps. Circles mark the pixel used in the histogram."

slope_error_max = max(analytical_slope.std.data.max(), numerical_slope.std.data.max())
fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
analytical_slope.estimate.plot(
    ax=axes[0, 0], cmap="terrain", cbar_title="Slope (°)", title="Original slope",
)
analytical_slope.std.plot(
    ax=axes[1, 0], cmap="Purples", vmin=0, vmax=slope_error_max,
    cbar_title="Slope error (1σ, °)", title="Analytical slope error",
)
numerical_slope.std.plot(
    ax=axes[1, 1], cmap="Purples", vmin=0, vmax=slope_error_max,
    cbar_title="Slope error (1σ, °)", title="Numerical slope error",
)
numerical_slope.samples.plot.hist(
    ax=axes[0, 1], title="Slope simulations at the marked pixel", xlabel="Slope (°)",
    ylabel="Number of simulations", legend=False,
)

# Locate the simulated pixel on each map
pixel_x, pixel_y = detail_dem.ij2xy(15, 15)
for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
    ax.plot(pixel_x, pixel_y, "o", markersize=6, markerfacecolor="none", markeredgecolor="black")

# Mark mean ±1σ at the selected pixel
pixel_mean = float(numerical_slope.mean.data[15, 15])
pixel_sigma = float(numerical_slope.std.data[15, 15])
axes[0, 1].errorbar(
    pixel_mean, 0.55, xerr=pixel_sigma, fmt="o", color="black", capsize=4,
    transform=axes[0, 1].get_xaxis_transform(), zorder=3,
)
axes[0, 1].text(
    pixel_mean, 0.62, f"±1σ = ±{pixel_sigma:.2f}°", ha="center",
    transform=axes[0, 1].get_xaxis_transform(),
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
)
plt.show()
```

See {ref}`terrain-attributes` for the terrain methods.

(uncertainty-coreg)=
### Coregistration fit

We can also propagate elevation errors **during a coregistration fit**, accounting for errors in either DEM or both.

In practice, this means assessing the uncertainty on the six **fitted translations and rotations in X/Y/Z**.
This is particularly useful to estimate the impact of limited static surfaces or slope/aspect distributions on the alignment.

**Analytical** propagation uses linearized parameter covariance propagation for {ref}`LZD <lzd>` from Hugonnet et al. (to-be-communicated)
and point-to-plane {ref}`ICP <icp>` from [Bonnabel et al. (2016)](https://doi.org/10.1109/ACC.2016.7526532).

**Numerical** propagation generates multiple {ref}`random error fields <uncertainty-simulation>`
to perturb the DEMs and repeat the coregistration fit, as for other methods.

Below, we examplify this propagation with an {ref}`LZD <lzd>` coregistration, on a small proportion of terrain:

```{code-cell} ipython3
# Select a small window on stable terrain
coreg_ref = ref.icrop((100, 100, 196, 196))
coreg_dem = tba.crop(coreg_ref)
coreg_errors = {"ref": es, "tba": es}
coreg_predictors = {
    name: {"slope": window.slope(), "max_curvature": window.max_curvature()}
    for name, window in {"ref": coreg_ref, "tba": coreg_dem}.items()
}
coreg_method = xdem.coreg.LZD(subsample=200, max_iterations=50)
coreg_options = {
    "inputs": {"coreg": {"reference_elev": coreg_ref}},
    "operation_options": {"coreg_method": coreg_method},
    "error_structure": coreg_errors,
    "error_predictors": coreg_predictors,
    "fit_kwargs": {"error_structure": coreg_errors, "error_predictors": coreg_predictors},
    "random_state": 42,
}

analytical_summary = xdem.uncertainty.propagate(
    coreg_dem, "coreg", method="analytical", **coreg_options,
)
numerical_summary = xdem.uncertainty.propagate(
    coreg_dem, "coreg", method="numerical", **coreg_options,
    n_simulations=50, return_samples=True, on_error="raise",
    random_field_options={"backend": "gstools"},
)
```

Using {meth}`~xdem.uncertainty.PropagationSummary.plot` recognizes the coregistration outputs, and displays **coregistration uncertainty as ellipses**,
with separate panels for translations and rotations, and each axis for horizontal and vertical components:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
analytical_summary.plot(axes=axes, color="C0", label="Analytical")
numerical_summary.plot(axes=axes, color="C1", label="Numerical (50 simulations)")
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    legend = dict(zip(labels, handles))
    ax.legend(legend.values(), legend.keys())
plt.show()
```

The above plotting often suffices because horizontal components often have a similar error magnitude, which differs from vertical
ones.
If you suspect horizontal directional effects due to your topography (e.g., limited orientations), inspect the 6 parameters directly!

Each entry shows {attr}`~geoutils.PropagationSummary.mean` ± {attr}`~geoutils.PropagationSummary.std` (1σ):

```{code-cell} ipython3
parameter_values = {}
for name, summary in {"Analytical": analytical_summary, "Numerical": numerical_summary}.items():
    values = pd.DataFrame({"mean": summary.mean, "std": summary.std})

    # Express rotations in microradians
    values.loc[["rx", "ry", "rz"]] *= 1e6
    parameter_values[name] = [
        f"{mean:.3g} ± {std:.3g}" for mean, std in zip(values["mean"], values["std"])
    ]

parameter_comparison = pd.DataFrame(parameter_values, index=[
    "Translation X (m)", "Translation Y (m)", "Translation Z (m)",
    "Rotation X (µrad)", "Rotation Y (µrad)", "Rotation Z (µrad)",
])
print(parameter_comparison.to_string())
```

(uncertainty-advanced)=
## Advanced usage

The following sections present more advanced usage of the uncertainty features, such as **fitting custom models**
for magnitude/correlation.

(uncertainty-custom-magnitude)=
### Custom magnitude fit

One can pass **a custom function** to fit the error magnitude bins, either as
`magnitude_model` and `magnitude_fit_kwargs` arguments to {meth}`~xdem.DEM.estimate_error_structure`, or to
{meth}`~xdem.ErrorStructure.refit_magnitude` more directly:

```{code-cell} ipython3
def slope_spread(slope, offset, quadratic):
    """Calculate error magnitude as a quadratic function of slope in degrees."""
    return offset + quadratic * slope**2


quadratic_errors = slope_errors.refit_magnitude(
    slope_spread,
    fit_kwargs={"p0": [1, 0.001], "bounds": (0, np.inf)},
)
quadratic_errors.plot_magnitude(xlabel="Slope (°)", ylabel="Elevation error (1σ, m)", title="Quadratic slope fit")
plt.show()
```

(uncertainty-custom-variogram)=
### Custom variogram fit

One can pass **a custom function** to fit the variogram lag bins, either as a component's `correlation` with
`fit_kwargs` in {meth}`~xdem.DEM.estimate_error_structure`, or to
{meth}`~xdem.ErrorStructure.refit_variogram` more directly:

```{code-cell} ipython3
def stretched_variogram(distance, effective_range, partial_sill, exponent):
    """Calculate semivariance with a fitted exponent controlling the correlation decrease."""
    return partial_sill * (1 - np.exp(-3 * (distance / effective_range)**exponent))


initial_range = single_range_errors.components["correlated"].correlation.effective_range
custom_correlation = single_range_errors.refit_variogram(
    stretched_variogram,
    fit_kwargs={
        "p0": [initial_range, 1, 1],
        "bounds": [(1, float("inf")), (0, float("inf")), (0.2, 2)],
    },
)
ax = custom_correlation.plot_correlation(kind="correlation")
plt.show()
```

Note that passing **bounds** and **initial values** for model parameters, as shown above, can help the fit converge!

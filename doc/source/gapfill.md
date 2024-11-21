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
(interpolation)=
# Gap-filling

xDEM contains routines to gap-fill elevation data or elevation differences depending on the type of terrain.

```{important}
Most of the approaches below are application-specific (e.g., glaciers) and might be moved to a separate package
in future releases.
```

So far, xDEM has three types of gap-filling methods:

- Inverse-distance weighting interpolation,
- Local hypsometric interpolation (only relevant for elevation differences and glacier applications),
- Regional hypsometric interpolation (also for glaciers).

The last two methods are described in [McNabb et al. (2019)](https://doi.org/10.5194/tc-13-895-2019).

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9  # Default 10 is a bit too big for coregistration plots
```

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example data"
:  code_prompt_hide: "Hide the code for opening example data"

from datetime import datetime

import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

import xdem

# Load a reference DEM from 2009
dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
# Load a DEM from 1990
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
# Load glacier outlines from 1990.
glaciers_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
glaciers_2010 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines_2010"))

# Make a dictionary of glacier outlines where the key represents the associated date.
outlines = {
    datetime(1990, 8, 1): glaciers_1990,
    datetime(2009, 8, 1): glaciers_2010,
}

# Cropping DEMs to a smaller extent to visualize the gap-filling better
bounds = (dem_2009.bounds.left, dem_2009.bounds.bottom,
          dem_2009.bounds.left + 200 * dem_2009.res[0], dem_2009.bounds.bottom + 150 * dem_2009.res[1])
dem_2009 = dem_2009.crop(bounds)
dem_1990 = dem_1990.crop(bounds)
```

We create a difference of DEMs object {class}`xdem.ddem.dDEM` to experiment on:

```{code-cell} ipython3
ddem = xdem.dDEM(raster=dem_2009 - dem_1990, start_time=datetime(1990, 8, 1), end_time=datetime(2009, 8, 1))

# The example DEMs are void-free, so let's make some random voids.
# Introduce a fifth of nans randomly throughout the dDEM.
mask = np.zeros_like(ddem.data, dtype=bool)
mask.ravel()[(np.random.choice(ddem.data.size, int(ddem.data.size/5), replace=False))] = True
ddem.set_mask(mask)
```

## Inverse-distance weighting interpolation

Inverse-distance weighting (IDW) interpolation of elevation differences is arguably the simplest approach: voids are filled by a weighted-mean of the surrounding pixels values, with weight inversely proportional to their distance to the void pixel.

```{code-cell} ipython3
ddem_idw = ddem.interpolate(method="idw")
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

ddem_idw = ddem.copy(new_array=ddem_idw)

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before IDW\ngap-filling")
ddem.plot(cmap='RdYlBu', vmin=-20, vmax=20, ax=ax[0])
ax[1].set_title("After IDW\ngap-filling")
ddem_idw.plot(cmap='RdYlBu', vmin=-20, vmax=20, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

## Local hypsometric interpolation

This approach assumes that there is a relationship between the elevation and the elevation change in the dDEM, which is often the case for glaciers.
Elevation change gradients in late 1900s and 2000s on glaciers often have the signature of large thinning in the lower parts, while the upper parts might be less negative, or even positive.
This relationship is strongly correlated for a specific glacier, and weakly correlated on regional scales.
With the local (glacier specific) hypsometric approach, elevation change gradients are estimated for each glacier separately.
This is simply a linear or polynomial model estimated with the dDEM and a reference DEM.
Then, voids are interpolated by replacing them with what "should be there" at that elevation, according to the model.

```{code-cell} ipython3
ddem_localhyps = ddem.interpolate(method="local_hypsometric", reference_elevation=dem_2009, mask=glaciers_1990)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

ddem_localhyps = ddem.copy(new_array=ddem_localhyps)

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before local\nhypsometric\ngap-filling")
ddem.plot(cmap='RdYlBu', vmin=-20, vmax=20, ax=ax[0])
ax[1].set_title("After local\nhypsometric\ngap-filling")
ddem_localhyps.plot(cmap='RdYlBu', vmin=-20, vmax=20, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

Where the binning can be visualized as such:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code for hypsometric binning"
:  code_prompt_hide: "Hide code for hypsometric binning"

dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
outlines_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

ddem = xdem.dDEM(dem_2009 - dem_1990, start_time=np.datetime64("1990-08-01"), end_time=np.datetime64("2009-08-01"))

ddem.data /= 2009 - 1990

scott_1990 = outlines_1990.query("NAME == 'Scott Turnerbreen'")
mask = scott_1990.create_mask(ddem)

ddem_bins = xdem.volume.hypsometric_binning(ddem[mask], dem_2009[mask])
stds = xdem.volume.hypsometric_binning(ddem[mask], dem_2009[mask], aggregation_function=np.std)

plt.figure(figsize=(8, 8))
plt.grid(zorder=0)
plt.plot(ddem_bins["value"], ddem_bins.index.mid, linestyle="--", zorder=1)

plt.barh(
    y=ddem_bins.index.mid,
    width=stds["value"],
    left=ddem_bins["value"] - stds["value"] / 2,
    height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
    zorder=2,
    edgecolor="black",
)
for bin in ddem_bins.index:
    plt.vlines(ddem_bins.loc[bin, "value"], bin.left, bin.right, color="black", zorder=3)

plt.xlabel("Elevation change (m / a)")
plt.twiny()
plt.barh(
    y=ddem_bins.index.mid,
    width=ddem_bins["count"] / ddem_bins["count"].sum(),
    left=0,
    height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
    zorder=2,
    alpha=0.2,
)
plt.xlabel("Normalized area distribution (hypsometry)")

plt.ylabel("Elevation (m a.s.l.)")

plt.tight_layout()
plt.show()
```

*Caption: Hypsometric elevation change of Scott Turnerbreen on Svalbard from 1990--2009. The width of the bars indicate the standard deviation of the bin. The light blue background bars show the area distribution with elevation.*

## Regional hypsometric interpolation

Similarly to local hypsometric interpolation, the elevation change is assumed to be largely elevation-dependent.
With the regional approach (often also called "global"), elevation change gradients are estimated for all glaciers in an entire region, instead of estimating one by one.

```{code-cell} ipython3
ddem.set_mask(mask)
ddem_reghyps = ddem.interpolate(method="regional_hypsometric", reference_elevation=dem_2009, mask=glaciers_1990)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

ddem_reghyps = ddem.copy(new_array=ddem_reghyps)

# Plot before and after
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before regional\nhypsometric\ngap-filling")
ddem.plot(cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[0])
ax[1].set_title("After regional\nhypsometric\ngap-filling")
ddem_reghyps.plot(cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[1], cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code for hypsometric binning"
:  code_prompt_hide: "Hide code for hypsometric binning"

mask = outlines_1990.create_mask(ddem)

ddem_bins = xdem.volume.hypsometric_binning(ddem[mask], dem_2009[mask])
stds = xdem.volume.hypsometric_binning(ddem[mask], dem_2009[mask], aggregation_function=np.std)

plt.figure(figsize=(8, 8))
plt.grid(zorder=0)


plt.plot(ddem_bins["value"], ddem_bins.index.mid, linestyle="--", zorder=1)

plt.barh(
    y=ddem_bins.index.mid,
    width=stds["value"],
    left=ddem_bins["value"] - stds["value"] / 2,
    height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
    zorder=2,
    edgecolor="black",
)
for bin in ddem_bins.index:
    plt.vlines(ddem_bins.loc[bin, "value"], bin.left, bin.right, color="black", zorder=3)

plt.xlabel("Elevation change (m / a)")
plt.twiny()
plt.barh(
    y=ddem_bins.index.mid,
    width=ddem_bins["count"] / ddem_bins["count"].sum(),
    left=0,
    height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
    zorder=2,
    alpha=0.2,
)
plt.xlabel("Normalized area distribution (hypsometry)")
plt.ylabel("Elevation (m a.s.l.)")

plt.tight_layout()
plt.show()
```
*Caption: Regional hypsometric elevation change in central Svalbard from 1990--2009. The width of the bars indicate the standard deviation of the bin. The light blue background bars show the area distribution with elevation.*

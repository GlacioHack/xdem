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
# Differencing and volume change

**Example data**

Example data in this chapter are loaded as follows:

```{code-cell} ipython3
from datetime import datetime

import geoutils as gu
import numpy as np

import xdem

# Load a reference DEM from 2009
dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"), datetime=datetime(2009, 8, 1))
# Load a DEM from 1990
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"), datetime=datetime(1990, 8, 1))
# Load glacier outlines from 1990.
glaciers_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
glaciers_2010 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines_2010"))

# Make a dictionary of glacier outlines where the key represents the associated date.
outlines = {
    datetime(1990, 8, 1): glaciers_1990,
    datetime(2009, 8, 1): glaciers_2010,
}
```

## dDEM interpolation

There are many approaches to interpolate a dDEM.
A good comparison study for glaciers is McNabb et al., ([2019](https://doi.org/10.5194/tc-13-895-2019)).
So far, xDEM has three types of interpolation:

- Linear spatial interpolation
- Local hypsometric interpolation
- Regional hypsometric interpolation

Let's first create a {class}`xdem.ddem.dDEM` object to experiment on:

```{code-cell} ipython3
ddem = xdem.dDEM(raster=dem_2009 - dem_1990, start_time=dem_1990.datetime, end_time=dem_2009.datetime)

# The example DEMs are void-free, so let's make some random voids.
# Introduce 50000 nans randomly throughout the dDEM.
mask = np.zeros_like(ddem.data, dtype=bool)
mask.ravel()[(np.random.choice(ddem.data.size, 50000, replace=False))] = True
ddem.set_mask(mask)
```

### Linear spatial interpolation

Linear spatial interpolation (also often called bilinear interpolation) of dDEMs is arguably the simplest approach: voids are filled by a an average of the surrounding pixels values, weighted by their distance to the void pixel.

```{code-cell} ipython3
ddem.interpolate(method="linear")
```

```{eval-rst}
.. plot:: code/comparison_plot_spatial_interpolation.py

```

### Local hypsometric interpolation

This approach assumes that there is a relationship between the elevation and the elevation change in the dDEM, which is often the case for glaciers.
Elevation change gradients in late 1900s and 2000s on glaciers often have the signature of large melt in the lower parts, while the upper parts might be less negative, or even positive.
This relationship is strongly correlated for a specific glacier, and weakly correlated on regional scales (see [Regional hypsometric interpolation]).
With the local (glacier specific) hypsometric approach, elevation change gradients are estimated for each glacier separately.
This is simply a linear or polynomial model estimated with the dDEM and a reference DEM.
Then, voids are interpolated by replacing them with what "should be there" at that elevation, according to the model.

```{code-cell} ipython3
ddem.interpolate(method="local_hypsometric", reference_elevation=dem_2009, mask=glaciers_1990)
```

```{eval-rst}
.. plot:: code/comparison_plot_local_hypsometric_interpolation.py

```

*Caption: The elevation dependent elevation change of Scott Turnerbreen on Svalbard from 1990--2009. The width of the bars indicate the standard deviation of the bin. The light blue background bars show the area distribution with elevation.*

### Regional hypsometric interpolation

Similarly to [Local hypsometric interpolation], the elevation change is assumed to be largely elevation-dependent.
With the regional approach (often also called "global"), elevation change gradients are estimated for all glaciers in an entire region, instead of estimating one by one.
This is advantageous in respect to areas where voids are frequent, as not even a single dDEM value has to exist on a glacier in order to reconstruct it.
Of course, the accuracy of such an averaging is much lower than if the local hypsometric approach is used (assuming it is possible).

```{code-cell} ipython3
ddem.interpolate(method="regional_hypsometric", reference_elevation=dem_2009, mask=glaciers_1990)
```

```{eval-rst}
.. plot:: code/comparison_plot_regional_hypsometric_interpolation.py

```

*Caption: The regional elevation dependent elevation change in central Svalbard from 1990--2009. The width of the bars indicate the standard deviation of the bin. The light blue background bars show the area distribution with elevation.*

## The DEMCollection object

Keeping track of multiple DEMs can be difficult when many different extents, resolutions and CRSs are involved, and {class}`xdem.demcollection.DEMCollection` is xDEM's answer to make this simple.
We need metadata on the timing of these products.
The DEMs can be provided with the `datetime=` argument upon instantiation, or the attribute could be set later.
Multiple outlines are provided as a dictionary in the shape of `{datetime: outline}`.

```{eval-rst}
.. minigallery:: xdem.DEMCollection
        :add-heading:
```

[See here for the outline filtering syntax](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).

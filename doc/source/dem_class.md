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
(dem-class)=

# The digital elevation model ({class}`~xdem.DEM`)

Below, a summary of the {class}`~xdem.DEM` object and its methods.

(dem-obj-def)=

## Object definition and attributes

A {class}`~xdem.DEM` is a {class}`~geoutils.Raster` with an additional georeferenced vertical dimension stored in the attribute {attr}`~xdem.DEM.vcrs`.
It inherits the **four main attributes** of {class}`~geoutils.Raster` which are {attr}`~xdem.DEM.data`,
{attr}`~xdem.DEM.transform`, {attr}`~xdem.DEM.crs` and {attr}`~xdem.DEM.nodata`.

Many other useful raster attributes and methods are available through the {class}`~geoutils.Raster` object, such as
{attr}`~geoutils.Raster.bounds`, {attr}`~geoutils.Raster.res`, {func}`~xdem.DEM.reproject` and {func}`~xdem.DEM.crop` .

```{tip}
The complete list of {class}`~geoutils.Raster` attributes and methods can be found in [GeoUtils' API](https://geoutils.readthedocs.io/en/stable/api.html#raster) and more info on rasters on [GeoUtils' Raster documentation page](https://geoutils.readthedocs.io/en/stable/raster_class.html).
```

## Open and save

A {class}`~xdem.DEM` is opened by instantiating with either a {class}`str`, a {class}`pathlib.Path`, a {class}`rasterio.io.DatasetReader` or a
{class}`rasterio.io.MemoryFile`, as for a {class}`~geoutils.Raster`.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 400
pyplot.rcParams['savefig.dpi'] = 400
```

```{code-cell} ipython3
import xdem

# Instantiate a DEM from a filename on disk
filename_dem = xdem.examples.get_path("longyearbyen_ref_dem")
dem = xdem.DEM(filename_dem)
dem
```

Detailed information on the {class}`~xdem.DEM` is printed using {func}`~geoutils.Raster.info`, along with basic statistics using `stats=True`:

```{code-cell} ipython3
# Print details of raster
dem.info(stats=True)
```

```{important}
The {class}`~xdem.DEM` data array remains implicitly unloaded until {attr}`~xdem.DEM.data` is called. For instance, here, calling {class}`~xdem.DEM.info()` with `stats=True` automatically loads the array in-memory.

The georeferencing metadata ({attr}`~xdem.DEM.transform`, {attr}`~xdem.DEM.crs`, {attr}`~xdem.DEM.nodata`), however, is always loaded. This allows to pass it effortlessly to other objects requiring it for geospatial operations (reproject-match, rasterizing a vector, etc).
```

A {class}`~xdem.DEM` is saved to file by calling {func}`~xdem.DEM.save` with a {class}`str` or a {class}`pathlib.Path`.

```{code-cell} ipython3
# Save raster to disk
dem.save("mydem.tif")
```
```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("mydem.tif")
```

## Plotting

Plotting a DEM is done using {func}`~xdem.DEM.plot`, and can be done alongside a vector file.

```{code-cell} ipython3
# Open a vector file of glacier outlines near the DEM
import geoutils as gu
fn_glacier_outlines = xdem.examples.get_path("longyearbyen_glacier_outlines")
vect_gla = gu.Vector(fn_glacier_outlines)

# Crop outlines to those intersecting the DEM
vect_gla = vect_gla.crop(dem)

# Plot the DEM and the vector file
dem.plot(cmap="terrain", cbar_title="Elevation (m)")
vect_gla.plot(dem, ec="k", fc="none")  # We pass the DEM as reference for the plot CRS
```

## Vertical referencing

The vertical reference of a {class}`~xdem.DEM` is stored in {attr}`~xdem.DEM.vcrs`, and derived either from its
{attr}`~xdem.DEM.crs` (if 3D) or assessed from the DEM product name during instantiation.

```{code-cell} ipython3
# Check vertical CRS of dataset
dem.vcrs
```

In this case, the DEM has no defined vertical CRS, which is quite common. To set the vertical CRS manually,
use {class}`~xdem.DEM.set_vcrs`. Then, to transform into another vertical CRS, use {class}`~xdem.DEM.to_vcrs`.

```{code-cell} ipython3
# Define the vertical CRS as the 3D ellipsoid of the 2D CRS
dem.set_vcrs("Ellipsoid")
# Transform to the EGM96 geoid
dem.to_vcrs("EGM96")
```

```{note}
For more details on vertical referencing, see the {ref}`vertical-ref` page.
```

## Terrain attributes

A wide range of terrain attributes can be derived from a {class}`~xdem.DEM`, with several methods and options available,
by calling the function corresponding to the attribute name such as {func}`~xdem.DEM.slope`.

```{code-cell} ipython3
# Derive slope using the Zevenberg and Thorne (1987) method
slope = dem.slope(method="ZevenbergThorne")
slope.plot(cmap="Reds", cbar_title="Slope (°)")
```

```{note}
For the full list of terrain attributes, see the {ref}`terrain-attributes` page.
```

## Statistics
The [`get_stats()`](https://geoutils.readthedocs.io/en/latest/gen_modules/geoutils.Raster.get_stats.html) method allows to extract key statistical information from a raster in a dictionary.

- Get all statistics in a dict:
```{code-cell} ipython3
dem.get_stats()
```

The DEM statistics functionalities in `xdem` are based on those in `geoutils`.
For more information on computing statistics, please refer to the [`geoutils` documentation](https://geoutils.readthedocs.io/en/latest/raster_class.html#obtain-statistics).


Note: as [`get_stats()`](https://geoutils.readthedocs.io/en/latest/gen_modules/geoutils.Raster.get_stats.html) is a raster method, it can also be used for terrain attributes:
```{code-cell} ipython3
slope.get_stats()
```

## Coregistration

3D coregistration is performed with {func}`~xdem.DEM.coregister_3d`, which aligns the
{class}`~xdem.DEM` to another DEM using a pipeline defined with a {class}`~xdem.coreg.Coreg`
object (defaults to horizontal and vertical shifts).

```{code-cell} ipython3
# Another DEM to-be-aligned to the first one
filename_tba = xdem.examples.get_path("longyearbyen_tba_dem")
dem_tba = xdem.DEM(filename_tba)

# Coregister (horizontal and vertical shifts)
dem_tba_coreg = dem_tba.coregister_3d(dem, xdem.coreg.NuthKaab() + xdem.coreg.VerticalShift())

# Plot the elevation change of the DEM due to coregistration
dh_tba = dem_tba - dem_tba_coreg.reproject(dem_tba, silent=True)
dh_tba.plot(cmap="Spectral", cbar_title="Elevation change due to coreg (m)")
```

```{note}
For more details on building coregistration pipelines and accessing metadata, see the {ref}`coregistration` page.
```

## Uncertainty analysis

Estimation of DEM-related uncertainty can be performed with {func}`~xdem.DEM.estimate_uncertainty`, which estimates both
**an error map** considering variable per-pixel errors and **the spatial correlation of errors**. The workflow relies
on an independent elevation data object that is **assumed to have either much finer or similar precision**, and uses
stable terrain as a proxy.

```{code-cell} ipython3
# Estimate elevation uncertainty assuming both DEMs have similar precision
sig_dem, rho_sig = dem.estimate_uncertainty(dem_tba_coreg, precision_of_other="same", random_state=42)

# The error map variability is estimated from slope and curvature by default
sig_dem.plot(cmap="Purples", cbar_title=r"Random error in elevation (1$\sigma$, m)")

# The spatial correlation function represents how much errors are correlated at a certain distance
print("Elevation errors at a distance of 1 km are correlated at {:.2f} %.".format(rho_sig(1000) * 100))
```

```{note}
We use `random_state` to ensure a fixed randomized output. It is **only necessary if you need your results to be exactly reproductible**.

For more details on quantifying random and structured errors, see the {ref}`uncertainty` page.
```

## Cropping a DEM

The DEM cropping functionalities in `xdem` are based on those in `geoutils` ([`crop()`](https://geoutils.readthedocs.io/en/latest/gen_modules/geoutils.Raster.crop.html#geoutils.Raster.crop), [`icrop()`](https://geoutils.readthedocs.io/en/latest/gen_modules/geoutils.Raster.icrop.html#geoutils.Raster.icrop)).
For more information on using cropping functions, please refer to the [`geoutils` documentation](https://geoutils.readthedocs.io/en/latest/raster_class.html#crop).

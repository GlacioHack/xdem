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

Many other useful raster attributes, such as {attr}`~xdem.DEM.bounds` and {attr}`~xdem.DEM.res` and raster methods 
such {attr}`~xdem.DEM.reproject` or {attr}`~xdem.DEM.crop` are available through the {class}`~geoutils.Raster` object. 

```{important}
Below, we only cover a few core aspects linked to the {class}`~geoutils.Raster` object. For more details, see [GeoUtils' Raster documentation page](https://geoutils.readthedocs.io/en/stable/raster_class.html).
The complete list of {class}`~geoutils.Raster` attributes and methods in [GeoUtils' API](https://geoutils.readthedocs.io/en/stable/api.html#raster).
```

## Open and save

A {class}`~xdem.DEM` is opened, as for a {class}`~geoutils.Raster`, by instantiating with either a {class}`str`, a {class}`pathlib.Path`, a {class}`rasterio.io.DatasetReader` or a
{class}`rasterio.io.MemoryFile`.


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

```{note}
Calling {class}`~xdem.DEM.info()` with `stats=True` automatically loads the array in-memory, like any other operation calling {attr}`~xdem.DEM.data`.
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




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
(elevation-point-cloud)=

# The elevation point cloud ({class}`~xdem.EPC`)

An elevation point cloud can be used with many features of xDEM (vertical referencing, coregistration and bias-corrections, uncertainty analysis) 
but not for methods requiring continuous gridded data (terrain attributes).

Below, a summary of the {class}`~xdem.EPC` object and its methods.

(dem-obj-def)=

## Object definition and attributes

An {class}`~xdem.EPC` is a {class}`~geoutils.PointCloud` with an additional georeferenced vertical dimension stored in the attribute {attr}`~xdem.EPC.vcrs`.
It inherits the **main attribute** of {class}`~geoutils.PointCloud` which is a geodataframe {attr}`~xdem.EPC.ds`.

Other useful point cloud attributes and methods are available through the {class}`~geoutils.PointCloud` object, such as
{attr}`~xdem.EPC.point_count`, {func}`~xdem.EPC.grid` and {func}`~xdem.EPC.subsample` .

```{tip}
The complete list of {class}`~geoutils.PointCloud` attributes and methods can be found in [GeoUtils' API](https://geoutils.readthedocs.io/en/stable/api.html#point-cloud) and more info on rasters on [GeoUtils' Point cloud documentation page](https://geoutils.readthedocs.io/en/stable/pointcloud_class.html).
```

Through GeoUtils, xDEM supports the reading and writing of point clouds both from vector-type files (e.g., ESRI shapefile, geopackage,
geoparquet) usually used for **sparse point clouds**, and from point-cloud-type files (e.g., LAS, LAZ, COPC) usually
used for **dense point clouds**.

```{warning}
Support for LAS files is still preliminary and loads all data in memory during data-related operations. We are working on adding operations with chunked reading.
```

## Open and save

An {class}`~xdem.EPC` is opened by instantiating the class with a {class}`str`, a {class}`pathlib.Path`, a {class}`geopandas.GeoDataFrame`,
a {class}`geopandas.GeoSeries` or a {class}`shapely.Geometry`

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 400
pyplot.rcParams['savefig.dpi'] = 400
```

```{code-cell} ipython3
import xdem

# Instantiate an EPC from a filename on disk
filename_epc = xdem.examples.get_path("longyearbyen_ref_dem")
dem = xdem.EPC(filename_epc)
dem
```

Detailed information on the {class}`~xdem.EPC` is printed using {func}`~geoutils.EPC.info`, along with basic statistics using `stats=True`:

```{code-cell} ipython3
# Print details of elevation point cloud
epc.info(stats=True)
```

```{important}
For a LAS/LAZ file, the {class}`~xdem.EPC` arrays remain implicitly unloaded until {attr}`~xdem.EPC.ds` is called. For instance, here, calling {class}`~xdem.EPC.info()` with `stats=True` automatically loads the array in-memory.

The metadata ({attr}`~xdem.EPC.point_count`, {attr}`~xdem.EPC.crs`, {attr}`~xdem.EPC.bounds`), however, is always loaded. This allows to pass it effortlessly to other objects requiring it for geospatial operations (reproject-match, rasterizing a vector, etc).
```

An {class}`~xdem.EPC` is saved to file by calling {func}`~xdem.EPC.save` with a {class}`str` or a {class}`pathlib.Path`.

```{code-cell} ipython3
# Save elevation point cloud to disk
epc.save("myepc.gpkg")
```
```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("myepc.gpkg")
```

## Plotting

Plotting an EPC is done using {func}`~xdem.EPC.plot`, and can be done alongside a raster or vector file.

```{code-cell} ipython3
# Open a vector file of glacier outlines near the EPC
import geoutils as gu
fn_glacier_outlines = xdem.examples.get_path("longyearbyen_glacier_outlines")
vect_gla = gu.Vector(fn_glacier_outlines)

# Crop outlines to those intersecting the DEM
vect_gla = vect_gla.crop(epc)

# Plot the DEM and the vector file
epc.plot(cmap="terrain", cbar_title="Elevation (m)")
vect_gla.plot(epc, ec="k", fc="none")  # We pass the EPC as reference for the plot CRS
```

## Vertical referencing

The vertical reference of an {class}`~xdem.EPC` is stored in {attr}`~xdem.EPC.vcrs`, and derived either from its
{attr}`~xdem.EPC.crs` (if 3D) or assessed from the DEM product name or user-input during instantiation.

```{code-cell} ipython3
# Check vertical CRS of dataset
epc.vcrs
```

In this case, the EPC has no defined vertical CRS, which is quite common. To set the vertical CRS manually,
use {class}`~xdem.EPC.set_vcrs`. Then, to transform into another vertical CRS, use {class}`~xdem.EPC.to_vcrs`.

```{code-cell} ipython3
# Define the vertical CRS as the 3D ellipsoid of the 2D CRS
epc.set_vcrs("Ellipsoid")
# Transform to the EGM96 geoid
epc.to_vcrs("EGM96")
```

```{note}
For more details on vertical referencing, see the {ref}`vertical-ref` page.
```

## Statistics
The {func}`~gu.PointCloud.get_stats` method allows to extract key statistical information from a point cloud in a dictionary.

- Get all statistics in a dict:
```{code-cell} ipython3
epc.get_stats()
```

The DEM statistics functionalities in xDEM are based on those in GeoUtils.
For more information on computing statistics, please refer to [GeoUtils' documentation](https://geoutils.readthedocs.io/en/stable/stats.html).

## Coregistration

3D coregistration is performed with {func}`~xdem.EPC.coregister_3d`, which aligns the
{class}`~xdem.EPC` to another DEM or EPC using a pipeline defined with a {class}`~xdem.coreg.Coreg`
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

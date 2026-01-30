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
(epc-class)=

# The elevation point cloud ({class}`~xdem.EPC`)

Below, a summary of the {class}`~xdem.EPC` object and its methods.

(epc-obj-def)=

## Object definition and attributes

An {class}`~xdem.EPC` is a {class}`~geoutils.PointCloud` with an additional georeferenced vertical dimension stored in the attribute {attr}`~xdem.EPC.vcrs`.
It can be used with many features of xDEM (vertical referencing, coregistration and bias-corrections, uncertainty analysis)
but not for methods requiring continuous gridded data (terrain attributes).

The {class}`~xdem.EPC` inherits the **main attribute** of {class}`~geoutils.PointCloud` which is a geodataframe {attr}`~xdem.EPC.ds`.
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

An {class}`~xdem.EPC` is opened by instantiating the class with a {class}`str`, a {class}`pathlib.Path`, or a {class}`geopandas.GeoDataFrame`,
or a {class}`geopandas.GeoSeries`, containing either only 2D or only 3D point geometries.
In the following example, our file contains 2D geometries, and so we need to pass a **data column name** to associate to the elevation value.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 400
pyplot.rcParams['savefig.dpi'] = 400
```

```{code-cell} ipython3
:tags: [hide-output]

import xdem

# Instantiate a point cloud from a filename on disk, passing the relevant data column
filename_epc = xdem.examples.get_path("longyearbyen_epc")
# For this ICESat-2 file, the elevation column name is "h_li"
epc = xdem.EPC(filename_epc, data_column="h_li")
epc
```

Detailed information on the {class}`~xdem.EPC` is printed using {func}`~xdem.EPC.info`, along with basic statistics using `stats=True`:

```{code-cell} ipython3
# Print details of elevation point cloud
epc.info()
```

```{important}
For a LAS/LAZ file, the {class}`~xdem.EPC` arrays remain implicitly unloaded until {attr}`~xdem.EPC.ds` is called. For instance, here, calling {class}`~xdem.EPC.info()` with `stats=True` automatically loads the array in-memory.

The metadata ({attr}`~xdem.EPC.point_count`, {attr}`~xdem.EPC.crs`, {attr}`~xdem.EPC.bounds`), however, is always loaded. This allows to pass it effortlessly to other objects requiring it for geospatial operations (reproject-match, rasterizing a vector, etc).
```

An {class}`~xdem.EPC` is saved to file by calling {func}`~xdem.EPC.to_file` with a {class}`str` or a {class}`pathlib.Path`.

```{code-cell} ipython3
# Save elevation point cloud to disk
epc.to_file("myepc.gpkg")
```
```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("myepc.gpkg")
```

## Plotting

Plotting an elevation point cloud is done using {func}`~xdem.EPC.plot`, and can be done alongside a raster or vector file.

```{code-cell} ipython3
# Open a vector file of glacier outlines near the EPC
import geoutils as gu
fn_glacier_outlines = xdem.examples.get_path("longyearbyen_glacier_outlines")
vect_gla = gu.Vector(fn_glacier_outlines)

# Reproject to local projected CRS
epc = epc.reproject(crs=epc.get_metric_crs())

# Crop outlines to those intersecting the EPC
vect_gla = vect_gla.crop(epc)

# Plot the DEM and the vector file
epc.plot(cmap="terrain", markersize=0.5, cbar_title="Elevation (m)")
vect_gla.plot(epc, ec="k", fc="none")  # We pass the EPC as reference for the plot CRS
```

## Vertical referencing

The vertical reference of an {class}`~xdem.EPC` is stored in {attr}`~xdem.EPC.vcrs`, and derived either from its
{attr}`~xdem.EPC.crs` (if 3D) or assessed from the DEM product name or user-input during instantiation.

```{code-cell} ipython3
# Check vertical CRS of dataset
epc.vcrs
```

In this case, the elevation point cloud is using a 3D CRS extended from the ellipsoid. To override manually with another vertical CRS,
{class}`~xdem.EPC.set_vcrs` is used. Then, to transform into another vertical CRS, use {class}`~xdem.EPC.to_vcrs`.

```{code-cell} ipython3
# Transform to the EGM96 geoid
epc = epc.to_vcrs("EGM96")
```

```{note}
For more details on vertical referencing, see the {ref}`vertical-ref` page.
```

## Statistics
The {func}`~xdem.EPC.get_stats` method allows to extract statistical information from a point cloud in a dictionary.

- Get all statistics in a dict:
```{code-cell} ipython3
epc.get_stats()
```

The point cloud statistics functionalities in xDEM are based on those in GeoUtils.
For more information on computing statistics, please refer to [GeoUtils' documentation](https://geoutils.readthedocs.io/en/stable/stats.html).

## Coregistration

3D coregistration is performed with {func}`~xdem.EPC.coregister_3d`, which aligns the
{class}`~xdem.EPC` to a {class}`~xdem.DEM` (or vice versa) using a pipeline defined with a {class}`~xdem.coreg.Coreg`
object (defaults to horizontal and vertical shifts).

```{important}
Coregistration in xDEM currently support only EPC–DEM or DEM–DEM inputs, because coregistering two sparse EPCs is rarely useful and
dense EPCs (like lidar point clouds) reach similar coregistration accuracy when converted to a DEM (e.g. using {func}`~xdem.EPC.grid`).
```

```{code-cell} ipython3
# A reference DEM to the elevation point cloud
filename_ref = xdem.examples.get_path("longyearbyen_ref_dem")
dem_ref = xdem.DEM(filename_ref)
epc = epc.reproject(dem_ref)

# Coregister (horizontal and vertical shifts)
epc_aligned = epc.coregister_3d(dem_ref, xdem.coreg.LZD())

# Plot the elevation change before/after coregistration over a hillshade
hs = dem_ref.hillshade()
dh_before = epc - dem_ref.interp_points(epc)
dh_after = epc_aligned - dem_ref.interp_points(epc_aligned)

# Remove outliers
# dh_before[gu.stats.nmad(dh_before) < 3] = np.nan
# dh_after[gu.stats.nmad(dh_after) < 3] = np.nan

import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before\ncoregistration")
hs.plot(ax=ax[0], cmap="Greys_r", add_cbar=False)
dh_before.plot(ax=ax[0], cmap='RdYlBu', vmin=-10, vmax=10, markersize=0.5, cbar_title="Elevation differences (m)")
ax[1].set_title("After\ncoregistration")
hs.plot(ax=ax[1], cmap="Greys_r", add_cbar=False)
dh_after.plot(ax=ax[1], cmap='RdYlBu', vmin=-10, vmax=10, markersize=0.5, cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

```{note}
For more details on building coregistration pipelines and accessing metadata, see the {ref}`coregistration` page.
```

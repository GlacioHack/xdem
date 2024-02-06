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
(quick-start)=

# Quick start

A short example show-casing some main functionalities of xDEM.
To find an example about a specific functionality, jump to {ref}`quick-gallery` right below.

## Short example

```{note}
:class: margin
xDEM relies largely on [its sister-package GeoUtils](https://geoutils.readthedocs.io/) for geospatial handling 
(reprojection, cropping, raster-vector interface, point interpolation) as well as numerics 
(NumPy interface). 🙂
```

The package functionalities revolve around the {class}`~xdem.DEM` class (a subclass of {class}`~geoutils.Raster`), from 
which most methods can be called and {class}`~xdem.coreg.Coreg` classes to build coregistration pipelines.

Below, in a few lines, we load a raster and a vector, crop them to a common extent, re-assign raster values around
a buffer of the vector, perform calculations on the modified raster, and finally plot and save it!

```{code-cell} ipython3
import xdem
import geoutils as gu

# Examples files: paths to two GeoTIFFs and an ESRI shapefile of glacier outlines
filename_dem_ref = xdem.examples.get_path("longyearbyen_ref_dem")
filename_dem_tba = xdem.examples.get_path("longyearbyen_tba_dem")
filename_glacier_outlines = xdem.examples.get_path("longyearbyen_glacier_outlines")

# Open files by instantiating DEM and Vector
# (DEMs are loaded lazily = only metadata but not array unless required)
dem_ref = xdem.DEM(filename_dem_ref)
dem_tba = xdem.DEM(filename_dem_tba)
vect_gla = gu.Vector(filename_glacier_outlines)

# Clip outlines to extent of reference DEM (method from GeoUtils)
vect_gla = vect_gla.crop(dem_ref, clip=True)

# Create a mask from glacier polygons (method from GeoUtils)
mask_gla = vect_gla.create_mask(dem_ref)

# We convert the vertical CRS of one DEM to the EGM96 geoid
dem_ref.to_vcrs("EGM96", force_source_vcrs="Ellipsoid")

# Align the two DEMs with a coregistration: 3D shift + 2nd-order 2D poly
mycoreg = xdem.coreg.NuthKaab() + xdem.coreg.Deramp(poly_order=2)
mycoreg.fit(dem_ref, dem_tba, inlier_mask=~mask_gla)
dem_aligned = mycoreg.apply(dem_tba)

# Get elevation difference
dh = dem_ref - dem_aligned

# Derive slope and curvature attributes
slope, maximum_curvature = xdem.terrain.get_terrain_attribute(
    dem_ref, attribute=["slope", "maximum_curvature"]
)

# Estimate elevation change error from stable terrain as a function of slope and curvature 
dem_error = xdem.spatialstats.infer_heteroscedasticity_from_stable(
    dh, list_var=[slope, maximum_curvature], unstable_mask=mask_gla
)[0]

# Plot error map of elevation difference and glacier outlines
dem_error.show(vmin=2, vmax=7, cmap="Reds", cbar_title=r"Elevation error (1$\sigma$, m)")
vect_gla.show(dem_error, fc='none', ec='k', lw=0.5)

# Save to file
dem_error.save("my_dem_error.tif")
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("my_dem_error.tif")
```

(quick-gallery)=
## More examples

To dive into more illustrated code, explore our gallery of examples that is composed of:
- An {ref}`examples-basic` section using the simpler features in xDEM (terrain, pre-defined coregistration and uncertainty pipelines),
- An {ref}`examples-advanced` section using advanced features (detailed coregistration and uncertainty analyses).

See also the concatenated list of examples below.

```{eval-rst}
.. minigallery:: xdem.DEM
    :add-heading: Examples using DEMs
```

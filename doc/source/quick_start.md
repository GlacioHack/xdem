---
file_format: mystnb
mystnb:
  execution_timeout: 60
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

Below is a short example show-casing some of the core functionalities of xDEM.
To find an example about a specific functionality, jump directly to {ref}`quick-gallery`.

## Download data examples

Example data from xDEM is automatically downloaded when calling `xdem.example.get_all_data()` as shown below.

See the {ref}`data-example` page to learn more about all our example data and different download options.

```{code-cell} ipython3
import xdem

output_dir = xdem.examples.get_all_data()
```

## Python example

```{note}
:class: margin
xDEM relies largely on [its sister-package GeoUtils](https://geoutils.readthedocs.io/) for geospatial handling
(reprojection, cropping, raster-vector interface, point interpolation) as well as numerics
(NumPy interface). 🙂
```

xDEM revolves around the {class}`~xdem.DEM` class (a subclass of {class}`~geoutils.Raster`) and the {class}`~xdem.EPC` class (a subclass of {class}`~geoutils.PointCloud`), from
which most methods can be called. The {class}`~xdem.coreg.Coreg` class is used to build modular coregistration pipelines.

Below, in a few lines, we load two DEMs and a vector of glacier outlines, crop them to a common extent,
align the DEMs using coregistration, estimate the elevation change, estimate elevation change error using stable
terrain, and finally plot and save the result!


```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
```

```{code-cell} ipython3
import xdem
import geoutils as gu

# Examples files: filenames of two DEMs and some glacier outlines
fn_dem_ref = xdem.examples.get_path("longyearbyen_ref_dem")
fn_dem_tba = xdem.examples.get_path("longyearbyen_tba_dem")
fn_glacier_outlines = xdem.examples.get_path("longyearbyen_glacier_outlines")

# Print filenames
print(f"DEM 1: {fn_dem_ref}, \nDEM 2: {fn_dem_tba}, \nOutlines: {fn_glacier_outlines}")
```

```{tip}
:class: margin
Set up your {ref}`verbosity` to manage outputs to the console (or a file) during execution!
```

```{code-cell} ipython3
# Open files by instantiating DEM and Vector
# (DEMs are loaded lazily = only metadata but not array unless required)
dem_ref = xdem.DEM(fn_dem_ref)
dem_tba = xdem.DEM(fn_dem_tba)
vect_gla = gu.Vector(fn_glacier_outlines)

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
slope, max_curvature = xdem.terrain.get_terrain_attribute(
    dem_ref, attribute=["slope", "max_curvature"]
)

# Estimate elevation change error from stable terrain as a function of slope and curvature
dh_err = xdem.spatialstats.infer_heteroscedasticity_from_stable(
    dh, list_var=[slope, max_curvature], unstable_mask=mask_gla
)[0]

# Plot dh, glacier outlines and its error map
dh.plot(cmap="RdYlBu", cbar_title="Elevation change (m)")
vect_gla.plot(dh, fc='none', ec='k', lw=0.5)

dh_err.plot(ax="new", vmin=2, vmax=7, cmap="Reds", cbar_title=r"Elevation change error (1$\sigma$, m)")
vect_gla.plot(dh_err, fc='none', ec='k', lw=0.5)

# Save to file
dh_err.to_file("dh_error.tif")
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("dh_error.tif")
```

## Command-line example

xDEM also support several **workflows** (series of analysis steps) through a command line interface, such as the **accuracy** and **topo** workflows to perform an accuracy assessment or generate a topographical summary of your elevation data, respectively.
These workflows use inputs on disk (elevation data files and a YAML configuration file), and generate outputs on disk (plots, tables, and HTML/PDF report).

```{caution}
The command-line interface for workflows is experimental, and its API might change rapidly.
```

The configuration file should contain, at minima, the path to elevation files but aims to be modular to tune the use of xDEM's features (see details in the **{ref}`cli` reference page**).
Here is an example of configuration file for the **accuracy** workflow:

```{literalinclude} _workflows/accuracy_config.yaml
:language: yaml
```

Then, run the workflow using:

```shell
xdem accuracy --config config_file.yaml
```

The outputs are written by default in an `outputs` folder of the current directory.

(quick-gallery)=
## More examples

To dive into more illustrated code, explore our gallery of examples that is composed of:
- A {ref}`examples-basic` section on simpler routines (terrain attributes, pre-defined coregistration and uncertainty pipelines),
- An {ref}`examples-advanced` section using advanced pipelines (for in-depth coregistration and uncertainty analysis).

See also the concatenated list of examples below.

```{eval-rst}
.. minigallery:: xdem.DEM
    :add-heading: Examples elevation data
```

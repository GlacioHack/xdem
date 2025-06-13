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
(terrain-attributes)=

# Terrain attributes

xDEM can derive a wide range of **terrain attributes** from a DEM.

Attributes are derived in pure Python for modularity (e.g., varying window size) and other uses (e.g., uncertainty),
and tested for consistency against [gdaldem](https://gdal.org/programs/gdaldem.html) and [RichDEM](https://richdem.readthedocs.io/).

## Quick use

Terrain attribute methods can be derived directly from a {class}`~xdem.DEM`, using for instance {func}`xdem.DEM.slope`.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
```

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the opening of example files."
:  code_prompt_hide: "Hide the opening of example files."

import xdem

# Open a DEM from a filename on disk
filename_dem = xdem.examples.get_path("longyearbyen_ref_dem")
dem = xdem.DEM(filename_dem)
```

```{code-cell} ipython3
# Slope from DEM method
slope = dem.slope()
# Or from terrain module using an array input
slope = xdem.terrain.slope(dem.data, resolution=dem.res)
```

```{tip}
All attributes can be derived using either SciPy or Numba as computing engine. Both engines perform similarly for attributes
based on a surface fit (e.g., slope, aspect, curvatures), while Numba is much faster for windowed indexes (e.g., TPI, roughness).

Note that Numba requires a [just-in-time compilation](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html)
at the first execution of an attribute (usually lasting about 5 seconds). This
compilation is cached and can later be re-used in the same Python environment.
```

## Summary of supported methods

```{list-table}
   :widths: 1 1 1
   :header-rows: 1
   :stub-columns: 1

   * - Attribute
     - Unit (if DEM in meters)
     - Reference
   * - {ref}`slope`
     - Degrees (default) or radians
     - [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) or [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)
   * - {ref}`aspect`
     - Degrees (default) or radians
     - [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) or [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)
   * - {ref}`hillshade`
     - Unitless
     - [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) or [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)
   * - {ref}`curv`
     - Meters{sup}`-1` * 100
     - [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107) and [Moore et al. (1991)](https://doi.org/10.1002/hyp.3360050103)
   * - {ref}`plancurv`
     - Meters{sup}`-1` * 100
     - [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)
   * - {ref}`profcurv`
     - Meters{sup}`-1` * 100
     - [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)
   * - {ref}`tpi`
     - Meters
     - [Weiss (2001)](http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf)
   * - {ref}`tri`
     - Meters
     - [Riley et al. (1999)](http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf) or [Wilson et al. (2007)](http://dx.doi.org/10.1080/01490410701295962)
   * - {ref}`roughness`
     - Meters
     - [Dartnell (2000)](https://environment.sfsu.edu/node/11292)
   * - {ref}`rugosity`
     - Unitless
     - [Jenness (2004)](<https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2>)
   * - {ref}`fractrough`
     - Fractal dimension number (1 to 3)
     - [Taud and Parrot (2005)](https://doi.org/10.4000/geomorphologie.622)
```

```{note}
Only grids with **equal pixel size in X and Y** are currently supported. Transform into such a grid with {func}`xdem.DEM.reproject`.
```

(slope)=
## Slope

{func}`xdem.DEM.slope`

The slope of a DEM describes the tilt, or gradient, of each pixel in relation to its neighbours.
It is most often described in degrees, where a flat surface is 0° and a vertical cliff is 90°.
No tilt direction is stored in the slope map; a 45° tilt westward is identical to a 45° tilt eastward.

The slope $\alpha$ can be computed either by the method of [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) (default)
based on a refined gradient formulation on a 3x3 pixel window, or by the method of
[Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107) based on a plane fit on a 3x3 pixel window.

For both methods, $\alpha = arctan(\sqrt{p^{2} + q^{2}})$ where $p$ and $q$ are the gradient components west-to-east and south-to-north, respectively, with for [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918):

$$
p_{\textrm{Horn}}=\frac{(h_{++} + 2h_{+0} + h_{+-}) - (h_{-+} + 2h_{-0} + h_{--})}{8 \Delta x},\\
q_{\textrm{Horn}}=\frac{(h_{++} + 2h_{0+} + h_{-+}) - (h_{+-} + 2h_{0-} + h_{--})}{8 \Delta y},
$$

and for [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107):

$$
p_{\textrm{ZevTho}} = \frac{h_{+0} - h_{-0}}{2 \Delta x},\\
q_{\textrm{ZevTho}} = \frac{h_{0+} - h_{0-}}{2 \Delta y},
$$

where $h_{ij}$ is the elevation at pixel $ij$, where indexes $i$ and $j$ correspond to east-west and north-south directions respectively,
and take values of either the center ($0$), west or south ($-$), or east or north ($+$):

```{list-table}
   :widths: 1 1 1 1
   :header-rows: 1
   :stub-columns: 1

   * -
     - West
     - Center
     - East
   * - North
     - $h_{-+}$
     - $h_{0+}$
     - $h_{++}$
   * - Center
     - $h_{-0}$
     - $h_{00}$
     - $h_{+0}$
   * - South
     - $h_{--}$
     - $h_{0-}$
     - $h_{+-}$
```


Finally, $\Delta x$
and $\Delta y$ correspond to the pixel resolution west-east and south-north, respectively.

The differences between methods are illustrated in the {ref}`sphx_glr_advanced_examples_plot_slope_methods.py`
example.

```{code-cell} ipython3
slope = dem.slope()
slope.plot(cmap="Reds", cbar_title="Slope (°)")
```

(aspect)=
## Aspect

{func}`xdem.DEM.aspect`

The aspect describes the orientation of strongest slope.
It is often reported in degrees, where a slope tilting straight north corresponds to an aspect of 0°, and an eastern
aspect is 90°, south is 180° and west is 270°. By default, a flat slope is given an arbitrary aspect of 180°.

The aspect $\theta$ is based on the same variables as the slope, and thus varies similarly between the method of
[Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) (default) and that of
[Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107):

$$
\theta = arctan(\frac{p}{q}),
$$

with $p$ and $q$ defined in the slope section.

```{warning}
A north aspect represents the upper direction of the Y axis in the coordinate reference system of the
input, {attr}`xdem.DEM.crs`, which might not represent the true north.
```

```{code-cell} ipython3
aspect = dem.aspect()
aspect.plot(cmap="twilight", cbar_title="Aspect (°)")
```

(hillshade)=
## Hillshade

{func}`xdem.DEM.hillshade`

The hillshade is a slope map, shaded by the aspect of the slope.
With a westerly azimuth (a simulated sun coming from the west), all eastern slopes are slightly darker.
This mode of shading the slopes often generates a map that is much more easily interpreted than the slope.

The hillshade $hs$ is directly based on the slope $\alpha$ and aspect $\theta$, and thus also varies between the method of [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) (default) and that of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107), and
is often scaled between 1 and 255:

$$
hs = 1 + 254 \left[ sin(alt) cos(\alpha) + cos(alt) sin(\alpha) sin(2\pi - azim - \theta) \right],
$$

where $alt$ is the shading altitude (90° = from above) and $azim$ is the shading azimuth (0° = north).

Note, however, that the hillshade is not a shadow map; no occlusion is taken into account so it does not represent "true" shading.
It therefore has little analytic purpose, but it still constitutes a great visualization tool.

```{code-cell} ipython3
hillshade = dem.hillshade()
hillshade.plot(cmap="Greys_r", cbar_title="Hillshade")
```

(curv)=
## Curvature

{func}`xdem.DEM.curvature`

The curvature is the second derivative of elevation, which highlights the convexity or concavity of the terrain.
If a surface is convex (like a mountain peak), it will have positive curvature.
If a surface is concave (like a through or a valley bottom), it will have negative curvature.
The curvature values in units of m{sup}`-1` are quite small, so they are by convention multiplied by 100.

The curvature $c$ is based on the method of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)
expanded to compute the surface laplacian in [Moore et al. (1991)](https://doi.org/10.1002/hyp.3360050103).

$$

c = - 100 \frac{(h_{+0} + h_{-0} + h_{0+} + h_{0-}) - 4 h_{00}}{\Delta x \Delta y}.

$$

```{code-cell} ipython3
curvature = dem.curvature()
curvature.plot(cmap="RdGy_r", cbar_title="Curvature (100 / m)", vmin=-1, vmax=1, interpolation="antialiased")
```

(plancurv)=
## Planform curvature

{func}`xdem.DEM.planform_curvature`

The planform curvature is the curvature perpendicular to the direction of slope, reported in 100 m{sup}`-1`, also based
on [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107):

$$

planc = -2\frac{DH² + EG² -FGH}{G²+H²}

$$

with:

$$

D &= \frac{h_{0+} + h_{0-} - 2h_{00}} {2\Delta y^{2}}, \\
E &= \frac{h_{+0} + h_{-0} - 2h_{00}} {2\Delta x^{2}},  \\
F &= \frac{h_{--} + h_{++} - h_{-+} - h_{+-}} {4 \Delta x \Delta y}, \\
G &= \frac{h_{0-} - h_{0+}}{2\Delta y}, \\
H &= \frac{h_{-0} - h_{+0}}{2\Delta x}.

$$

```{code-cell} ipython3
planform_curvature = dem.planform_curvature()
planform_curvature.plot(cmap="RdGy_r", cbar_title="Planform curvature (100 / m)", vmin=-1, vmax=1, interpolation="antialiased")
```

(profcurv)=
## Profile curvature

{func}`xdem.DEM.profile_curvature`

The profile curvature is the curvature parallel to the direction of slope, reported in 100 m{sup}`-1`, also based on
[Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107):

$$

profc = 2\frac{DG² + EH² + FGH}{G²+H²}

$$

based on the equations in the planform curvature section for $D$, $E$, $F$, $G$ and $H$.

```{code-cell} ipython3
profile_curvature = dem.profile_curvature()
profile_curvature.plot(cmap="RdGy_r", cbar_title="Profile curvature (100 / m)", vmin=-1, vmax=1, interpolation="antialiased")
```

(tpi)=
## Topographic position index

{func}`xdem.DEM.topographic_position_index`

The topographic position index (TPI) is a metric of slope position, described in [Weiss (2001)](http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf), that corresponds to the difference of the elevation of a central
pixel with the average of that of neighbouring pixels. Its unit is that of the DEM (typically meters) and it can be
computed for any window size (default 3x3 pixels).

$$
tpi = h_{00} - \textrm{mean}_{i\neq 0, j\neq 0} (h_{ij}) .
$$

```{code-cell} ipython3
tpi = dem.topographic_position_index()
tpi.plot(cmap="Spectral", cbar_title="Topographic position index (m)", vmin=-5, vmax=5)
```

(tri)=
## Terrain ruggedness index

{func}`xdem.DEM.terrain_ruggedness_index`

The terrain ruggedness index (TRI) is a metric of terrain ruggedness, based on cumulated differences in elevation between
a central pixel and its surroundings. Its unit is that of the DEM (typically meters) and it can be computed for any
window size (default 3x3 pixels).

For topography (default), the method of [Riley et al. (1999)](http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf) is generally used, where the TRI is computed by the squareroot of squared differences with
neighbouring pixels:

$$
tri_{\textrm{Riley}} = \sqrt{\sum_{ij}(h_{00} - h_{ij})^{2}}.
$$

For bathymetry, the method of [Wilson et al. (2007)](http://dx.doi.org/10.1080/01490410701295962) is generally used,
where the TRI is defined by the mean absolute difference with neighbouring pixels:

$$
tri_{\textrm{Wilson}} = \textrm{mean}_{ij} (|h_{00} - h{ij}|) .
$$

```{code-cell} ipython3
tri = dem.terrain_ruggedness_index()
tri.plot(cmap="Purples", cbar_title="Terrain ruggedness index (m)")
```

(roughness)=
## Roughness

{func}`xdem.DEM.roughness`

The roughness is a metric of terrain ruggedness, based on the maximum difference in elevation in the surroundings,
described in [Dartnell (2000)](https://environment.sfsu.edu/node/11292). Its unit is that of the DEM (typically meters) and it can be computed for any window size (default 3x3 pixels).

$$
r_{\textrm{D}} = \textrm{max}_{ij} (h{ij}) -  \textrm{min}_{ij} (h{ij}) .
$$

```{code-cell} ipython3
roughness = dem.roughness()
roughness.plot(cmap="Oranges", cbar_title="Roughness (m)")
```

(rugosity)=
## Rugosity

{func}`xdem.DEM.rugosity`

The rugosity is a metric of terrain ruggedness, based on the ratio between planimetric and real surface area,
described in [Jenness (2004)](<https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2>).
It is unitless, and is only supported for a 3x3 window size.

$$
r_{\textrm{J}} = \frac{\sum_{k \in [1,8]} A(T_{00, k})}{\Delta x \Delta y}.
$$

where $A(T_{00,k})$ is the area of one of the 8 triangles connected from the center of the center pixel $00$ to the
centers of its 8 neighbouring pixels $k$, cropped to intersect only the center pixel. This surface area is computed in three dimensions, accounting for elevation differences.

```{code-cell} ipython3
rugosity = dem.rugosity()
rugosity.plot(cmap="YlOrRd", cbar_title="Rugosity")
```

(fractrough)=
## Fractal roughness

{func}`xdem.DEM.fractal_roughness`

The fractal roughness is a metric of terrain ruggedness based on the local fractal dimension estimated by the volume
box-counting method of [Taud and Parrot (2005)](https://doi.org/10.4000/geomorphologie.622).
The fractal roughness is computed by estimating the fractal dimension in 3D space, for a local window centered on the
DEM pixels. Its unit is that of a dimension, and is always between 1 (dimension of a line in 3D space) and 3
(dimension of a cube in 3D space). It can only be computed on window sizes larger than 5x5 pixels, and defaults to 13x13.

```{code-cell} ipython3
fractal_roughness = dem.fractal_roughness()
fractal_roughness.plot(cmap="Reds", cbar_title="Fractal roughness (dimensions)")
```

## Generating attributes in multiprocessing
Computing terrain attributes over large digital elevation models can be computationally expensive,
especially for high-resolution datasets. To improve performance and reduce memory usage,
xDEM supports multiprocessing for out-of-memory attribute calculations using the `mp_config` parameter.
The resulting attribute is saved directly to disk under the filename specified in `mp_config`.

### Example
```{code-cell} ipython3
from geoutils.raster.distributed_computing import MultiprocConfig

mp_config = MultiprocConfig(chunk_size=200, outfile="hillshade.tif")
hillshade = dem.hillshade(mp_config=mp_config)
hillshade
```
```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("hillshade.tif")
```

## Generating multiple attributes at once

Often, one may seek more terrain attributes than one, e.g. both the slope and the aspect.
Since both are dependent on the gradient of the DEM, calculating them separately is computationally repetitive.
Multiple terrain attributes can be calculated from the same gradient using the {func}`xdem.DEM.get_terrain_attribute` function.

```{eval-rst}
.. minigallery:: xdem.terrain.get_terrain_attribute
```

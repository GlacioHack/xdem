(terrain-attributes)=

# Terrain attributes

xDEM can derive a wide range of **terrain attributes** from a DEM.

Attributes are derived in pure Python for modularity (e.g., varying window size) and other uses (e.g., uncertainty), 
and tested for consistency against [gdaldem](https://gdal.org/programs/gdaldem.html) and [RichDEM](https://richdem.readthedocs.io/). 

## Quick use

Terrain attribute methods can either be called directly from a {class}`~xdem.DEM` (e.g., {func}`xdem.DEM.slope`) or 
through the {class}`~xdem.terrain` module which allows array input. If computational performance 
is key, xDEM can rely on [RichDEM](https://richdem.readthedocs.io/) by specifying `use_richdem=True` for speed-up 
of its supported attributes (slope, aspect, curvature).

## Slope

{func}`xdem.DEM.slope`

The slope of a DEM describes the tilt, or gradient, of each pixel in relation to its neighbours.
It is most often described in degrees, where a flat surface is 0° and a vertical cliff is 90°.
No tilt direction is stored in the slope map; a 45° tilt westward is identical to a 45° tilt eastward.

The slope can be computed either by the method of [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) (default)
based on a refined gradient formulation on a 3x3 pixel window, or by the method of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107) based on a plane fit on a 3x3 pixel window.

The differences between methods are illustrated in the {ref}`sphx_glr_basic_examples_plot_terrain_attributes.py`
example.

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_001.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.slope
```

## Aspect

{func}`xdem.DEM.aspect`

The aspect describes the orientation of strongest slope.
It is often reported in degrees, where a slope tilting straight north corresponds to an aspect of 0°, and an eastern
aspect is 90°, south is 180° and west is 270°. By default, a flat slope is given an arbitrary aspect of 180°.

As the aspect is directly based on the slope, it varies between the method of [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) (default) and that of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107).

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_002.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.aspect
        :add-heading:
```

## Hillshade

{func}`xdem.DEM.hillshade`

The hillshade is a slope map, shaded by the aspect of the slope.
The slope map is a good tool to visualize terrain, but it does not distinguish between a mountain and a valley.
It may therefore be slightly difficult to interpret in mountainous terrain.
Hillshades are therefore often preferable for visualizing DEMs.
With a westerly azimuth (a simulated sun coming from the west), all eastern slopes are slightly darker.
This mode of shading the slopes often generates a map that is much more easily interpreted than the slope map.

As the hillshade is directly based on the slope and aspect, it varies between the method of [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) (default) and that of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107).

Note, however, that the hillshade is not a shadow map; no occlusion is taken into account so it does not represent "true" shading.
It therefore has little analytic purpose, but it still constitutes a great visualization tool.

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_003.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.hillshade
        :add-heading:
```

## Curvature

{func}`xdem.DEM.curvature`

The curvature map is the second derivative of elevation, which highlights the convexity or concavity of the terrain.
If a surface is convex (like a mountain peak), it will have positive curvature.
If a surface is concave (like a through or a valley bottom), it will have negative curvature.
The curvature values in units of m{sup}`-1` are quite small, so they are by convention multiplied by 100.

The curvature is based on the method of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107).

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_004.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.curvature
        :add-heading:
```

## Planform curvature

{func}`xdem.DEM.planform_curvature`

The planform curvature is the curvature perpendicular to the direction of slope, reported in 100 m{sup}`-1`.

It is based on the method of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107).

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_005.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.planform_curvature
        :add-heading:
```

## Profile curvature

{func}`xdem.DEM.profile_curvature`

The profile curvature is the curvature parallel to the direction of slope, reported in 100 m{sup}`-1`..

It is based on the method of [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107).

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_006.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.profile_curvature
        :add-heading:
```

## Topographic Position Index

{func}`xdem.DEM.topographic_position_index`

The Topographic Position Index (TPI) is a metric of slope position, based on the method of [Weiss (2001)](http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf) that corresponds to the difference of the elevation of a central
pixel with the average of that of neighbouring pixels. Its unit is that of the DEM (typically meters) and it can be
computed for any window size (default 3x3 pixels).

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_007.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.topographic_position_index
        :add-heading:
```

## Terrain Ruggedness Index

{func}`xdem.DEM.terrain_ruggedness_index`

The Terrain Ruggedness Index (TRI) is a metric of terrain ruggedness, based on cumulated differences in elevation between
a central pixel and its surroundings. Its unit is that of the DEM (typically meters) and it can be computed for any
window size (default 3x3 pixels).

For topography (default), the method of [Riley et al. (1999)](http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf) is generally used, where the TRI is computed by the squareroot of squared differences with
neighbouring pixels.

For bathymetry, the method of [Wilson et al. (2007)](http://dx.doi.org/10.1080/01490410701295962) is generally used,
where the TRI is defined by the mean absolute difference with neighbouring pixels

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_008.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.terrain_ruggedness_index
        :add-heading:
```

## Roughness

{func}`xdem.DEM.roughness`

The roughness is a metric of terrain ruggedness, based on the maximum difference in elevation in the surroundings.
The roughness is based on the method of [Dartnell (2000)](http://dx.doi.org/10.14358/PERS.70.9.1081). Its unit is that of the DEM (typically meters) and it can be computed for any window size (default 3x3 pixels).

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_009.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.roughness
        :add-heading:
```

## Rugosity

{func}`xdem.DEM.rugosity`

The rugosity is a metric of terrain ruggedness, based on the ratio between planimetric and real surface area. The
rugosity is based on the method of [Jenness (2004)](<https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2>).
It is unitless, and is only supported for a 3x3 window size.

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_010.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.rugosity
        :add-heading:
```

## Fractal roughness

{func}`xdem.DEM.fractal_roughness`

The fractal roughness is a metric of terrain ruggedness based on the local fractal dimension estimated by the volume
box-counting method of [Taud and Parrot (2005)](https://doi.org/10.4000/geomorphologie.622).
The fractal roughness is computed by estimating the fractal dimension in 3D space, for a local window centered on the
DEM pixels. Its unit is that of a dimension, and is always between 1 (dimension of a line in 3D space) and 3
(dimension of a cube in 3D space). It can only be computed on window sizes larger than 5x5 pixels, and defaults to 13x13.

```{image} basic_examples/images/sphx_glr_plot_terrain_attributes_011.png
:width: 600
```

```{eval-rst}
.. minigallery:: xdem.terrain.fractal_roughness
        :add-heading:
```

## Generating multiple attributes at once

Often, one may seek more terrain attributes than one, e.g. both the slope and the aspect.
Since both are dependent on the gradient of the DEM, calculating them separately is computationally repetitive.
Multiple terrain attributes can be calculated from the same gradient using the {func}`xdem.DEM.get_terrain_attribute` function.

```{eval-rst}
.. minigallery:: xdem.terrain.get_terrain_attribute
        :add-heading:
```

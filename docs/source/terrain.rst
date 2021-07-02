.. _terrain_attributes:

Terrain attributes
==================

For analytic and visual purposes, deriving certain attributes of a DEM may be required.
Some are useful for direct analysis, such as a slope map to differentiate features of different angles, while others, like the hillshade, are great tools for visualizing a DEM.

.. contents:: Contents
        :local:

Slope
-----
:func:`xdem.terrain.slope`

The slope map of a DEM describes the tilt (or gradient) of each pixel in relation to its neighbours.
It is most often described in degrees, where a flat surface is 0° and a vertical cliff is 90°.
No tilt direction is stored in the slope map; a 45° tilt westward is identical to a 45° tilt eastward.

.. minigallery:: xdem.terrain.slope
        :add-heading:

Aspect
------
:func:`xdem.terrain.aspect`

The aspect map of a DEM describes which direction the slope is tilting of each pixel in relation to its neighbours.
It is most often described in degrees, where a slope tilting straight north would have an aspect of 0°, an eastern aspect is 90°, south is 180° and west is 270°.

.. minigallery:: xdem.terrain.aspect
        :add-heading:

Hillshade
---------
:func:`xdem.terrain.hillshade`

The hillshade is a slope map, shaded by the aspect of the slope.
The slope map is a good tool to visualize terrain, but it does not distinguish between a mountain and a valley.
It may therefore be slightly difficult to interpret in mountainous terrain.
Hillshades are therefore often preferable for visualizing DEMs.
With a westerly azimuth (a simulated sun coming from the west), all eastern slopes are slightly darker.
This mode of shading the slopes often generates a map that is much more easily interpreted than the slope map.

Note, however, that the hillshade is not a shadow map; no occlusion is taken into account so it does not represent "true" shading.
It therefore has little analytic purpose, but it still a great tool for visualization.

.. minigallery:: xdem.terrain.hillshade
        :add-heading:

Curvature
---------
:func:`xdem.terrain.curvature`

The curvature map is the second derivative of elevation.
It highlights the convexity or convavity of the terrain, and has many both analytic and visual purposes.
If a surface is convex (like a mountain peak), it will have positive curvature.
If a surface is concave (like a trough or a valley bottom), it will have negative curvature.

Usually, the curvature values are quite small, so they are by convention multiplied by 100.
For analytic purposes, it may therefore be worth considering dividing the output by 100.

.. minigallery:: xdem.terrain.curvature
        :add-heading:

Planform curvature
------------------
:func:`xdem.terrain.planform_curvature`

TODO: Add text.

.. minigallery:: xdem.terrain.planform_curvature
        :add-heading:

Profile curvature
-----------------
:func:`xdem.terrain.profile_curvature`

TODO: Add text.

.. minigallery:: xdem.terrain.profile_curvature
        :add-heading:

Generating multiple attributes at once
--------------------------------------

Often, one may seek more terrain attributes than one, e.g. both the slope and the aspect.
Since both are dependent on the gradient of the DEM, calculating them separately is unneccesarily repetitive.
Multiple terrain attributes can be calculated from the same gradient using the :func:`xdem.terrain.get_terrain_attribute` function.

.. minigallery:: xdem.terrain.get_terrain_attribute
        :add-heading:

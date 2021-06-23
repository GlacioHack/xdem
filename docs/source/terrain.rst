Terrain attributes
==================

For analytic and visual purposes, deriving certain attributes of a DEM may be required.
Some are useful for direct analysis, such as a slope map to differentiate features of different angles, while others, like the hillshade, are great tools for visualizing a DEM.

.. contents:: Contents
        :local:

**Example data**

.. literalinclude:: code/terrain.py
        :lines: 2-4

Attribute types
^^^^^^^^^^^^^^^


Slope
*****
:func:`xdem.terrain.slope`

The slope map of a DEM describes the tilt (or gradient) of each pixel in relation to its neighbours.
It is most often described in degrees, where a flat surface is 0° and a vertical cliff is 90°.
No tilt direction is stored in the slope map; a 45° tilt westward is identical to a 45° tilt eastward.

.. literalinclude:: code/terrain.py
        :lines: 6

.. plot:: code/terrain_plot_slope.py


Aspect
******
:func:`xdem.terrain.aspect`

The aspect map of a DEM describes which direction the slope is tilting of each pixel in relation to its neighbours.
It is most often described in degrees, where a slope tilting straight north would have an aspect of 0°, an eastern aspect is 90°, south is 180° and west is 270°.

.. literalinclude:: code/terrain.py
        :lines: 8

.. plot:: code/terrain_plot_aspect.py


Hillshade
*********
:func:`xdem.terrain.hillshade`

The hillshade is a slope map, shaded by the aspect of the slope.
The slope map is a good tool to visualize terrain, but it does not distinguish between a mountain and a valley.
It may therefore be slightly difficult to interpret in mountainous terrain.
Hillshades are therefore often preferable for visualizing DEMs.
With a westerly azimuth (a simulated sun coming from the west), all eastern slopes are slightly darker.
This mode of shading the slopes often generates a map that is much more easily interpreted than the slope map.

Note, however, that the hillshade is not a shadow map; no occlusion is taken into account so it does not represent "true" shading.
It therefore has little analytic purpose, but it still a great tool for visualization.


.. literalinclude:: code/terrain.py
        :lines: 10

.. plot:: code/terrain_plot_hillshade.py


Generating multiple attributes at once
**************************************

Often, one may seek more terrain attributes than one, e.g. both the slope and the aspect.
Since both are dependent on the gradient of the DEM, calculating them separately is unneccesarily repetitive.
Multiple terrain attributes can be calculated from the same gradient using the :func:`xdem.terrain.get_terrain_attribute` function:

.. literalinclude:: code/terrain.py
        :lines: 12-16



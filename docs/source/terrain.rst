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

The slope can be computed either by the method of `Horn (1981) <http://dx.doi.org/10.1109/PROC.1981.11918>`_ (default)
based on a refined gradient formulation on a 3x3 pixel window, or by the method of `Zevenbergen and Thorne (1987)
<http://dx.doi.org/10.1002/esp.3290120107>`_ based on a plane fit on a 3x3 pixel window.

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_001.png
  :width: 600

.. minigallery:: xdem.terrain.slope
        :add-heading:

Aspect
------
:func:`xdem.terrain.aspect`

The aspect describes the orientation of strongest slope.
It is often reported in degrees, where a slope tilting straight north corresponds to an aspect of 0°, and an eastern
aspect is 90°, south is 180° and west is 270°.

As the aspect is directly based on the slope, it varies between the method of `Horn (1981) <http://dx.doi.org/10.
1109/PROC.1981.11918>`_ (default) and that of `Zevenbergen and Thorne (1987) <http://dx.doi.org/10.1002/esp.3290120107>`_.

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_002.png
  :width: 600

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


As the hillshade is directly based on the slope and aspect, it varies between the method of `Horn (1981) <http://dx.doi
.org/10.1109/PROC.1981.11918>`_ (default) and that of `Zevenbergen and Thorne (1987) <http://dx.doi.org/10.1002/esp.
3290120107>`_.

Note, however, that the hillshade is not a shadow map; no occlusion is taken into account so it does not represent "true" shading.
It therefore has little analytic purpose, but it still constitutes a great visualization tool.

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_003.png
  :width: 600

.. minigallery:: xdem.terrain.hillshade
        :add-heading:

Curvature
---------
:func:`xdem.terrain.curvature`

The curvature map is the second derivative of elevation, which highlights the convexity or concavity of the terrain.
If a surface is convex (like a mountain peak), it will have positive curvature.
If a surface is concave (like a trough or a valley bottom), it will have negative curvature.

The curvature is based on the method of `Zevenbergen and Thorne (1987) <http://dx.doi.org/10.1002/esp.3290120107>`_.

Usually, the curvature values are quite small, so they are by convention multiplied by 100.
For analytic purposes, it may therefore be worth considering dividing the output by 100.

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_004.png
  :width: 600

.. minigallery:: xdem.terrain.curvature
        :add-heading:

Planform curvature
------------------
:func:`xdem.terrain.planform_curvature`

The planform curvature is the curvature perpendicular to the direction of slope. It is based on the method of
`Zevenbergen and Thorne (1987) <http://dx.doi.org/10.1002/esp.3290120107>`_.

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_005.png
  :width: 600

.. minigallery:: xdem.terrain.planform_curvature
        :add-heading:

Profile curvature
-----------------
:func:`xdem.terrain.profile_curvature`

The profile curvature is the curvature parallel to the direction of slope. It is based on the method of
`Zevenbergen and Thorne (1987) <http://dx.doi.org/10.1002/esp.3290120107>`_.

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_006.png
  :width: 600

.. minigallery:: xdem.terrain.profile_curvature
        :add-heading:

Topographic Position Index
--------------------------
:func:`xdem.terrain.topographic_position_index`

The Topographic Position Index (TPI) is a metric of slope position, based on the method of `Weiss (2001) <http://www
.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf>`_ that corresponds to the difference of the elevation of a central
pixel with the average of that of neighbouring pixels. It can be computed for any window size (default 3x3 pixels).

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_007.png
  :width: 600

.. minigallery:: xdem.terrain.topographic_position_index
        :add-heading:

Terrain Ruggedness Index
------------------------
:func:`xdem.terrain.terrain_ruggedness_index`

The Terrain Ruggedness Index (TRI) is a metric of terrain ruggedness, based on cumulated differences in elevation between
a central pixel and its surroundings. It can be computed for any window size (default 3x3 pixels).

For topography (default), the method of `Riley et al. (1999) <http://download.osgeo.org/qgis/doc/reference-docs/Terrain_
Ruggedness_Index.pdf>`_ is generally used, where the TRI is computed by the squareroot of squared differences with
neighbouring pixels.

For bathymetry, the method of `Wilson et al. (2007) <http://dx.doi.org/10.1080/01490410701295962>`_ is generally used,
where the TRI is defined by the mean absolute difference with neighbouring pixels

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_008.png
  :width: 600

.. minigallery:: xdem.terrain.terrain_ruggedness_index
        :add-heading:

Roughness
---------
:func:`xdem.terrain.roughness`

The roughness is a metric of terrain ruggedness, based on the maximum difference in elevation in the surroundings.
The roughness is based on the method of `Dartnell (2000) <http://dx.doi.org/10.14358/PERS.70.9.
1081>`_. It can be computed for any window size (default 3x3 pixels).

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_009.png
  :width: 600

.. minigallery:: xdem.terrain.roughness
        :add-heading:

Rugosity
--------
:func:`xdem.terrain.rugosity`

The rugosity is a metric of terrain ruggedness, based on the ratio between planimetric and real surface area. The
rugosity is based on the method of `Jenness (2004) <https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2>`_.
It is only supported for a 3x3 window size.

.. image:: ../build/_images/sphx_glr_plot_terrain_attributes_010.png
  :width: 600

.. minigallery:: xdem.terrain.rugosity
        :add-heading:


Generating multiple attributes at once
--------------------------------------

Often, one may seek more terrain attributes than one, e.g. both the slope and the aspect.
Since both are dependent on the gradient of the DEM, calculating them separately is computationally repetitive.
Multiple terrain attributes can be calculated from the same gradient using the :func:`xdem.terrain.get_terrain_attribute` function.

.. minigallery:: xdem.terrain.get_terrain_attribute
        :add-heading:

Differencing and volume change
=================================

.. contents:: Contents 
   :local:

**Example data**


Example data in this chapter are loaded as follows:

.. literalinclude:: code/comparison.py
        :lines: 5-32


dDEM interpolation
------------------
There are many approaches to interpolate a dDEM.
A good comparison study for glaciers is McNabb et al., (`2019 <https://doi.org/10.5194/tc-13-895-2019>`_).
So far, ``xdem`` has three types of interpolation:

- Linear spatial interpolation
- Local hypsometric interpolation
- Regional hypsometric interpolation

Let's first create a :class:`xdem.ddem.dDEM` object to experiment on:

.. literalinclude:: code/comparison.py
        :lines: 51-60


Linear spatial interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Linear spatial interpolation (also often called bilinear interpolation) of dDEMs is arguably the simplest approach: voids are filled by a an average of the surrounding pixels values, weighted by their distance to the void pixel.


.. literalinclude:: code/comparison.py
        :lines: 64

.. plot:: code/comparison_plot_spatial_interpolation.py
        

Local hypsometric interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This approach assumes that there is a relationship between the elevation and the elevation change in the dDEM, which is often the case for glaciers.
Elevation change gradients in late 1900s and 2000s on glaciers often have the signature of large melt in the lower parts, while the upper parts might be less negative, or even positive.
This relationship is strongly correlated for a specific glacier, and weakly correlated on regional scales (see `Regional hypsometric interpolation`_).
With the local (glacier specific) hypsometric approach, elevation change gradients are estimated for each glacier separately.
This is simply a linear or polynomial model estimated with the dDEM and a reference DEM.
Then, voids are interpolated by replacing them with what "should be there" at that elevation, according to the model.


.. literalinclude:: code/comparison.py
        :lines: 68

.. plot:: code/comparison_plot_local_hypsometric_interpolation.py
        

*Caption: The elevation dependent elevation change of Scott Turnerbreen on Svalbard from 1990--2009. The width of the bars indicate the standard devation of the bin. The light blue background bars show the area distribution with elevation.*


Regional hypsometric interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Similarly to `Local hypsometric interpolation`_, the elevation change is assumed to be largely elevation-dependent.
With the regional approach (often also called "global"), elevation change gradients are estimated for all glaciers in an entire region, instead of estimating one by one.
This is advantageous in respect to areas where voids are frequent, as not even a single dDEM value has to exist on a glacier in order to reconstruct it.
Of course, the accuracy of such an averaging is much lower than if the local hypsometric approach is used (assuming it is possible).

.. literalinclude:: code/comparison.py
        :lines: 72

.. plot:: code/comparison_plot_regional_hypsometric_interpolation.py
        

*Caption: The regional elevation dependent elevation change in central Svalbard from 1990--2009. The width of the bars indicate the standard devation of the bin. The light blue background bars show the area distribution with elevation.*

The DEMCollection object
------------------------
Keeping track of multiple DEMs can be difficult when many different extents, resolutions and CRSs are involved, and :class:`xdem.demcollection.DEMCollection` is ``xdem``'s answer to make this simple.
We need metadata on the timing of these products.
The DEMs can be provided with the ``datetime=`` argument upon instantiation, or the attribute could be set later.
Multiple outlines are provided as a dictionary in the shape of ``{datetime: outline}``.


.. minigallery:: xdem.DEMCollection
        :add-heading:

`See here for the outline filtering syntax <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>`_.

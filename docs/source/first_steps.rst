.. _first_steps:

First steps
===========

In construction


Simple usage
------------

.. code-block:: python

        import xdem

        dem1 = xdem.DEM("path/to/first_dem.tif")
        dem2 = xdem.DEM("path/to/second_dem.tif")

        difference = dem1 - dem2
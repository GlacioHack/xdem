.. xdem documentation master file, created by
   sphinx-quickstart on Fri Mar 19 14:30:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xdem's documentation!
================================
xdem aims to make Digital Elevation Model (DEM) analysis easy.
Coregistration, subtraction (and volume measurements), and error statistics should be available to anyone with the correct input data.


Simple usage
==================

.. code-block:: python

        import xdem

        dem1 = xdem.DEM("path/to/first_dem.tif")
        dem2 = xdem.DEM("path/to/second_dem.tif")

        difference = dem1 - dem2



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   coregistration
   biascorr
   comparison
   spatialstats
   robuststats
   terrain
   auto_examples/index.rst
   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

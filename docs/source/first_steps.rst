.. _first_steps:

Quick start
===========

Sample data
-----------

*xdem* comes with some sample data that are used throughout this documentation to demonstrate the features. If not done already, the sample data can be downloaded with the command

.. code-block:: python

        xdem.examples.download_longyearbyen_examples(overwrite=False)
        
The dataset ids and paths can be found from 

.. code-block:: python

        xdem.examples.FILEPATHS_DATA

Load DEM data and calculate elevation difference
------------------------------------------------

A simple example on how to load raster data and run simple arithmetic operations such as subtraction, plotting the data and saving to file can be found in the example gallery:

.. minigallery:: plot_dem_subtraction

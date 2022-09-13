.. _quick_start:

Quick start
===========

Sample data
-----------

xDEM comes with some sample data that is used throughout this documentation to demonstrate the features. If not done already, the sample data can be downloaded with the command

.. code-block:: python

        xdem.examples.download_longyearbyen_examples()
        
The dataset keys and paths can be found from 

.. code-block:: python

        xdem.examples.FILEPATHS_DATA

Load DEMs and calculate elevation difference
------------------------------------------------

.. code-block:: python

  import xdem
  
  # Load data
  dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
  dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))

  # Calculate the difference
  ddem = dem_2009 - dem_1990

  # Plot
  ddem.show(cmap='coolwarm_r', vmin=-20, vmax=20, cb_title="Elevation change (m)")

  # Save to file
  ddem.save("temp.tif")

A detailed example on how to load raster data, reproject it to a different grid/projection, run simple arithmetic operations such as subtraction, plotting the data and saving to file can be found in the example gallery :ref:`sphx_glr_basic_examples_plot_dem_subtraction.py`.

..
   .. raw:: html

       <div class="sphx-glr-thumbcontainer" tooltip="DEM subtraction">

   .. only:: html

    .. figure:: /auto_examples/images/thumb/sphx_glr_plot_dem_subtraction_thumb.png
	:alt: DEM subtraction

	:ref:`sphx_glr_auto_examples_plot_dem_subtraction.py`

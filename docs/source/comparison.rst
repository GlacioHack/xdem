DEM subtraction and volume change
=================================

.. contents:: Contents 
   :local:

Subtracting rasters
^^^^^^^^^^^^^^^^^^^
Calculating the difference of DEMs (dDEMs) is theoretically simple, but can in practice often be time consuming.
In ``xdem``, the aim is to minimize or completely remove potential pitfalls in this analysis.

Let's assume we have two perfectly aligned DEMs, with the same shape, extent, resolution and coordinate referencesystem (CRS) as each other.
Calculating a dDEM would then be as simple as:

.. code-block:: python

        ddem_data = dem1.data - dem2.data
        # If we want to inherit the georeferencing information:
        ddem_raster = xdem.DEM.from_array(ddem_data, dem1.transform, dem1.crs)

But in practice, our two DEMs are most often not perfectly aligned, which is why we might need a helper function for this:

.. code-block:: python
        
        ddem_raster = xdem.spatial_tools.subtract_rasters(dem1, dem2)

So what does this magical function do?
First, the nonreference; ``dem2``, will be reprojected to fit the shape, extent, resolution and CRS of ``dem1``.
This behaviour can be switched by changing the default ``reference="first"`` to ``reference="second"``.
Cubic spline interpolation is used by default to resample the data, which is sometimes slow, but provides the most accurate resampling results.
This can be changed with the ``resampling_method=`` keyword, for example to ``"bilinear"`` or ``"nearest"`` (`see the rasterio docs for the full suite of options <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_).
Most often, ``xdem`` works with Masked Arrays, where the mask signifies cells to exclude (presumably areas of no data).
The ``subtract_rasters()`` function makes sure that the outgoing mask is the union of the two ingoing masks.

The DEMCollection object
^^^^^^^^^^^^^^^^^^^^^^^^
Keeping track of multiple DEMs can be difficult when many different extents, resolutions and CRSs are involved, and the ``DEMCollection`` is ``xdem``'s answer to make this simple.
Let's first load some example data:


.. code-block:: python

        import geoutils as gu
        import xdem

        # Download the necessary data. This may take a few minutes.
        xdem.examples.download_longyearbyen_examples(overwrite=False)

        # Load a reference DEM from 2009
        dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
        # Load a DEM from 1990
        dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
        # Load glacier outlines from 1990.
        glaciers_1990 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
        glaciers_2010 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines_2010"])

        # Fake a future DEM to have a time-series of three DEMs
        dem_2060 = dem_2009.copy()
        # Assume that all glacier values will be 30 m lower than in 2009
        dem_2060.data[glaciers_2010.create_mask(dem_2060) == 255] -= 30

We also need some metadata on the timing of these products.
The DEMs can be provided with the ``datetime=`` argument upon instantiation, or the attribute could be set later.
Multiple outlines are provided as a dictionary in the shape of ``{datetime: outline}``:


.. code-block:: python

        from datetime import datetime

        dem_1990.datetime = datetime(1990, 8, 1)
        dem_2009.datetime = datetime(2009, 8, 1)
        dem_2060.datetime = datetime(2060, 8, 1)

        outlines = {
                datetime(1990, 8, 1): glaciers_1990,
                datetime(2009, 8, 1): glaciers_2010
        }

        
Now that we have three DEMs and glacier outlines with known dates, we can create a collection from them:

.. code-block:: python

        dems = xdem.DEMCollection(
                [dem_1990, dem_2009, dem_2060],
                outlines=outlines,
                reference_dem=dem_2009
        )

Now, we can easily calculate the elevation or volume change between the DEMs, for example on the glacier Scott Turnerbreen:

.. code-block:: python

        dems.get_cumulative_series(kind="dh", outline_filter="NAME == 'Scott Turnerbreen'")

which will return a Pandas Series:

.. code-block:: python

        1990-08-01     0.000000
        2009-08-01   -13.379259
        2060-08-01   -43.379259       
        dtype: float64

`See here for the outline filtering syntax <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>`_.

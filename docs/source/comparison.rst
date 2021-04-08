DEM subtraction and volume change
=================================

.. contents:: Contents 
   :local:

**Example data**


Example data in this chapter are loaded as follows:

.. code-block:: python

        from datetime import datetime        

        import geoutils as gu
        import xdem

        # Download the necessary data. This may take a few minutes.
        xdem.examples.download_longyearbyen_examples(overwrite=False)

        # Load a reference DEM from 2009
        dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"], datetime=datetime(2009, 8, 1))
        # Load a DEM from 1990
        dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"], datetime=datetime(1990, 8, 1))
        # Load glacier outlines from 1990.
        glaciers_1990 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
        glaciers_2010 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines_2010"])
        
        # Make a dictionary of glacier outlines where the key represents the associated date.
        outlines = {
                datetime(1990, 8, 1): glaciers_1990,
                datetime(2009, 8, 1): glaciers_2010
        }

        # Fake a future DEM to have a time-series of three DEMs
        dem_2060 = dem_2009.copy()
        # Assume that all glacier values will be 30 m lower than in 2009
        dem_2060.data[glaciers_2010.create_mask(dem_2060) == 255] -= 30
        dem_2060.datetime.year = 2060

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
:func:`xdem.spatial_tools.subtract_rasters`

.. code-block:: python
        
        ddem_raster = xdem.spatial_tools.subtract_rasters(dem1, dem2)

So what does this magical function do?
First, the nonreference; ``dem2``, will be reprojected to fit the shape, extent, resolution and CRS of ``dem1``.
This behaviour can be switched by changing the default ``reference="first"`` to ``reference="second"``.
Cubic spline interpolation is used by default to resample the data, which is sometimes slow, but provides the most accurate resampling results.
This can be changed with the ``resampling_method=`` keyword, for example to ``"bilinear"`` or ``"nearest"`` (`see the rasterio docs for the full suite of options <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_).
Most often, ``xdem`` works with Masked Arrays, where the mask signifies cells to exclude (presumably areas of no data).
The ``subtract_rasters()`` function makes sure that the outgoing mask is the union of the two ingoing masks.

dDEM interpolation
^^^^^^^^^^^^^^^^^^
There are many approaches to interpolate a dDEM.
A good comparison study for glaciers is McNabb et al., (`2019 <https://doi.org/10.5194/tc-13-895-2019>`_).
So far, ``xdem`` has three types of interpolation:

- Linear spatial interpolation
- Local hypsometric interpolation
- Regional hypsometric interpolation

Let's first create a :class:`xdem.ddem.dDEM` object to experiment on:

.. code-block:: python

        ddem = xdem.dDEM(
                raster=xdem.spatial_tools.subtract_rasters(dem_2009, dem_1990),
                start_time=dem_1990.datetime,
                end_time=dem_2009.datetime
        )

        # The example DEMs are void-free, so let's make some random voids.
        ddem.data.mask = np.zeros_like(ddem.data, dtype=bool)  # Reset the mask
        # Introduce 50000 nans randomly throughout the dDEM.
        ddem.data.mask.ravel()[np.random.choice(ddem.data.size, 50000, replace=False)] = True



Linear spatial interpolation
****************************
Linear spatial interpolation (also often called bilinear interpolation) of dDEMs is arguably the simplest approach: voids are filled by a an average of the surrounding pixels values, weighted by their distance to the void pixel.

.. code-block:: python

        ddem.interpolate(method="linear")


.. plot::
        
        import xdem
        import numpy as np
        import geoutils as gu

        xdem.examples.download_longyearbyen_examples(overwrite=False)

        dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
        dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
        outlines_1990 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])

        ddem = xdem.dDEM(
                xdem.spatial_tools.subtract_rasters(dem_2009, dem_1990, resampling_method="nearest"),
                start_time=np.datetime64("1990-08-01"),
                end_time=np.datetime64("2009-08-01")
        )
        # The example DEMs are void-free, so let's make some random voids.
        ddem.data.mask = np.zeros_like(ddem.data, dtype=bool)  # Reset the mask
        # Introduce 50000 nans randomly throughout the dDEM.
        ddem.data.mask.ravel()[np.random.choice(ddem.data.size, 50000, replace=False)] = True

        ddem.interpolate(method="linear")

        ylim = (300, 100)
        xlim = (800, 1050)
        
        plt.figure(figsize=(8, 5))
        plt.subplot(121)
        plt.imshow(ddem.data.squeeze(), cmap="coolwarm_r", vmin=-50, vmax=50)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.axis("off")
        plt.title("dDEM with random voids")
        plt.subplot(122)
        plt.imshow(ddem.filled_data.squeeze(), cmap="coolwarm_r", vmin=-50, vmax=50)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.axis("off")
        plt.title("Linearly interpolated dDEM")

        
        plt.tight_layout()
        plt.show()

Local hypsometric interpolation
*******************************
This approach assumes that there is a relationship between the elevation and the elevation change in the dDEM, which is often the case for glaciers.
Elevation change gradients in late 1900s and 2000s on glaciers often have the signature of large melt in the lower parts, while the upper parts might be less negative, or even positive.
This relationship is strongly correlated for a specific glacier, and weakly correlated on regional scales (see `Regional hypsometric interpolation`_).
With the local (glacier specific) hypsometric approach, elevation change gradients are estimated for each glacier separately.
This is simply a linear or polynomial model estimated with the dDEM and a reference DEM.
Then, voids are interpolated by replacing them with what "should be there" at that elevation, according to the model.

.. code-block:: python
        
        ddem.interpolate(method="local_hypsometric", reference_elevation=dem_2009, mask=outlines_1990)


.. plot::
        
        import xdem
        import geoutils as gu
        import numpy as np
        import matplotlib.pyplot as plt
        
        xdem.examples.download_longyearbyen_examples(overwrite=False)

        dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
        dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
        outlines_1990 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])

        ddem = xdem.dDEM(
                xdem.spatial_tools.subtract_rasters(dem_2009, dem_1990, resampling_method="nearest"),
                start_time=np.datetime64("1990-08-01"),
                end_time=np.datetime64("2009-08-01")
        )

        ddem.data /= (2009 - 1990)

        scott_1990 = outlines_1990.query("NAME == 'Scott Turnerbreen'")
        mask = (scott_1990.create_mask(ddem) == 255).reshape(ddem.data.shape)

        ddem_bins = xdem.volume.hypsometric_binning(ddem.data[mask], dem_2009.data[mask])
        stds = xdem.volume.hypsometric_binning(ddem.data[mask], dem_2009.data[mask], aggregation_function=np.std)

        plt.figure(figsize=(8, 8))
        plt.grid(zorder=0)
        plt.plot(ddem_bins["value"], ddem_bins.index.mid, linestyle="--", zorder=1)

        plt.barh(
                y=ddem_bins.index.mid,
                width=stds["value"],
                left=ddem_bins["value"] - stds["value"] / 2,
                height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
                zorder=2,
                edgecolor="black",
        )
        for bin in ddem_bins.index:
                plt.vlines(ddem_bins.loc[bin, "value"], bin.left, bin.right, color="black", zorder=3)

        plt.xlabel("Elevation change (m / a)")
        plt.twiny()
        plt.barh(
                y=ddem_bins.index.mid,
                width=ddem_bins["count"] / ddem_bins["count"].sum(),
                left=0,
                height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
                zorder=2,
                alpha=0.2,
        )
        plt.xlabel("Normalized area distribution (hypsometry)")
        
        plt.ylabel("Elevation (m a.s.l.)")

        plt.tight_layout()
        plt.show()

*Caption: The elevation dependent elevation change of Scott Turnerbreen on Svalbard from 1990--2009. The width of the bars indicate the standard devation of the bin. The light blue background bars show the area distribution with elevation.*


Regional hypsometric interpolation
**********************************
Similarly to `Local hypsometric interpolation`_, the elevation change is assumed to be largely elevation-dependent.
With the regional approach (often also called "global"), elevation change gradients are estimated for all glaciers in an entire region, instead of estimating one by one.
This is advantageous in respect to areas where voids are frequent, as not even a single dDEM value has to exist on a glacier in order to reconstruct it.
Of course, the accuracy of such an averaging is much lower than if the local hypsometric approach is used (assuming it is possible).

.. code-block:: python
        
        ddem.interpolate(method="regional_hypsometric", reference_elevation=dem_2009, mask=outlines_1990)

.. plot::
        
        import xdem
        import geoutils as gu
        import numpy as np
        import matplotlib.pyplot as plt

        xdem.examples.download_longyearbyen_examples(overwrite=False)

        dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
        dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
        outlines_1990 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])

        ddem = xdem.dDEM(
                xdem.spatial_tools.subtract_rasters(dem_2009, dem_1990, resampling_method="nearest"),
                start_time=np.datetime64("1990-08-01"),
                end_time=np.datetime64("2009-08-01")
        )

        ddem.data /= (2009 - 1990)

        mask = (outlines_1990.create_mask(ddem) == 255).reshape(ddem.data.shape)

        ddem_bins = xdem.volume.hypsometric_binning(ddem.data[mask], dem_2009.data[mask])
        stds = xdem.volume.hypsometric_binning(ddem.data[mask], dem_2009.data[mask], aggregation_function=np.std)

        plt.figure(figsize=(8, 8))
        plt.grid(zorder=0)

        

        plt.plot(ddem_bins["value"], ddem_bins.index.mid, linestyle="--", zorder=1)

        plt.barh(
                y=ddem_bins.index.mid,
                width=stds["value"],
                left=ddem_bins["value"] - stds["value"] / 2,
                height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
                zorder=2,
                edgecolor="black",
        )
        for bin in ddem_bins.index:
                plt.vlines(ddem_bins.loc[bin, "value"], bin.left, bin.right, color="black", zorder=3)

        plt.xlabel("Elevation change (m / a)")
        plt.twiny()
        plt.barh(
                y=ddem_bins.index.mid,
                width=ddem_bins["count"] / ddem_bins["count"].sum(),
                left=0,
                height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
                zorder=2,
                alpha=0.2,
        )
        plt.xlabel("Normalized area distribution (hypsometry)")
        plt.ylabel("Elevation (m a.s.l.)")

        plt.tight_layout()
        plt.show()

*Caption: The regional elevation dependent elevation change in central Svalbard from 1990--2009. The width of the bars indicate the standard devation of the bin. The light blue background bars show the area distribution with elevation.*

The DEMCollection object
^^^^^^^^^^^^^^^^^^^^^^^^
Keeping track of multiple DEMs can be difficult when many different extents, resolutions and CRSs are involved, and :class:`xdem.demcollection.DEMCollection` is ``xdem``'s answer to make this simple.
We need metadata on the timing of these products.
The DEMs can be provided with the ``datetime=`` argument upon instantiation, or the attribute could be set later.
Multiple outlines are provided as a dictionary in the shape of ``{datetime: outline}``:


In the examples, we have three DEMs and glacier outlines with known dates, so we can create a collection from them:

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

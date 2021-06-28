.. _spatialstats:

Spatial statistics
==================

Spatial statistics, also referred to as geostatistics, are essential for the analysis of observations distributed in space.
To analyze DEMs, ``xdem`` integrates spatial statistics tools specific to DEMs based on recent literature, and with routines partly relying on `scikit-gstat <https://mmaelicke.github.io/scikit-gstat/index.html>`_.

The spatial statistics tools can be used to assess the precision of DEMs (see the definition of precision in :ref:`intro`), and in particular:
    - account for non-stationarities of elevation measurement errors (e.g., varying precision of DEMs with terrain slope),
    - quantify the spatial correlation of measurement errors in DEMs (e.g., native spatial resolution, instrument noise),
    - estimate robust errors for observations integrated in space (e.g., average or sum of samples),
    - propagate errors between spatial ensembles at different scales (e.g., sum of glacier volume changes).

More details below.

.. contents:: Contents 
   :local:

Assumptions for statistical inference in spatial statistics
***********************************************************

Spatial statistics are valid if the variable of interest verifies the assumption of stationarity of the 1:superscript:`st` and 2:superscript:`nd` orders.
That is, if the two following assumptions are verified:
    1. The mean of the variable of interest is stationary in space, i.e. constant over sufficiently large areas,
    2. The variance of the variable of interest is stationary in space, i.e. constant over sufficiently large areas.

A sufficiently large averaging area is an area expected to fit within the spatial domain studied.

In other words, for a reliable analysis, the DEM should:
    1. Not contain systematic biases that do not average to zero over sufficiently large distances (e.g., shifts, tilts), but can contain large-scale pseudo-periodic biases (e.g., along-track undulations),
    2. Not contain measurement errors that vary significantly.

Precision of a single DEM, or a difference of elevation data
************************************************************

TO COMPLETE LATER IN MORE DETAILS WITH: Hugonnet et al. (in prep)

To infer the precision of a DEM, it is compared against other elevation data.
If the other elevation data is known to be of higher-precision, one can assume that the analysis of differences will represent the precision of the rougher DEM.
Otherwise, the difference will describe the precision with significant measurement errors originating from both the DEM and the other dataset.

Stable terrain: proxy for infering DEM precision
************************************************

To infer the precision of a DEM over all terrain, the proxy typically utilized is the stable terrain (i.e. terrain that has not moved such as bare rock).

and after the removal of systematic biases to ensure an optimalaccuracy (see :ref:`intro`).




Metrics for DEM precision
*************************

Pixel-wise elevation measurement error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The


Spatially-integrated elevation measurement error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard error (SE) of a statistic is the standard deviation of the distribution of this statistic.
For spatially distributed samples, the standard error of the mean (SEM) is of great interest as it allows quantification of the error of a mean (or sum) of samples in space.

The standard error  :math:`\sigma_{\overline{dh}}` of the mean :math:`\overline{dh}` of elevation changes samples :math:`dh` is typically derived as:

.. math::

        \sigma_{\overline{dh}} = \frac{\sigma_{dh}}{\sqrt{N}},

where :math:`\sigma_{dh}` is the dispersion of the samples, and :math:`N` is the number of **independent** observations.

However, several issues arise to estimate the standard error of a mean of elevation observations samples:
    1. The dispersion :math:`\sigma_{dh}` cannot be estimated directly on changing terrain that we are usually interested in measuring (e.g., glacier, snow, forest).
    2. The dispersion :math:`\sigma_{dh}` typically shows important non-stationarities (e.g., an error 10 times as large on steep slopes than flat slopes).
    3. The number of samples :math:`N` is generally not equal to the number of sampled DEM pixels, as those are not independent in space and the Ground Sampling Distance of the DEM does not necessarily correspond to its effective resolution.

Note that the SE represents completely stochastic (random) errors, and is therefore not accounting for possible remaining systematic errors have been accounted for, e.g. using one or multiple :ref:`coregistration` approaches.


Methods for DEM precision estimation
************************************


Non-stationarity in elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO: Add this section based on Hugonnet et al. (in prep)


Multi-range spatial correlations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based Rolstad et al. (2009), Dehecq et al. (2020), Hugonnet et al. (in prep)

.. literalinclude:: code/spatialstats.py
        :lines: 26-27


.. plot:: code/spatialstats_empirical_vgm.py


.. plot:: code/spatialstats_model_vgm.py

Spatially integrated measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Rolstad et al. (2009), Hugonnet et al. (in prep)

Propagation of correlated errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Krige's relation (Webster & Oliver, 2007

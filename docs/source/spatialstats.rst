.. _spatialstats:

Spatial statistics
==================

Spatial statistics, also referred to as `geostatistics <https://en.wikipedia.org/wiki/Geostatistics>`_, are essential for the analysis of observations distributed in space.
To analyze DEMs, ``xdem`` integrates spatial statistics tools specific to DEMs described in recent literature,
in particular in `Rolstad et al. (2009) <https://doi.org/10.3189/002214309789470950>`_,
`Dehecq et al. (2020) <https://doi.org/10.3389/feart.2020.566802>`_ and
`Hugonnet et al. (2021) <https://doi.org/10.1038/s41586-021-03436-z>`_. The implementation of these methods relies
partly on the package `scikit-gstat <https://mmaelicke.github.io/scikit-gstat/index.html>`_.

The spatial statistics tools can be used to assess the precision of DEMs (see the definition of precision in :ref:`intro`).
In particular, these tools help to:

- account for non-stationarities of elevation measurement errors (e.g., varying precision of DEMs with terrain slope),
- quantify the spatial correlation of measurement errors in DEMs (e.g., native spatial resolution, instrument noise),
- estimate robust errors for observations integrated in space (e.g., average or sum of samples),
- propagate errors between spatial ensembles at different scales (e.g., sum of glacier volume changes).

.. contents:: Contents 
   :local:

Assumptions for statistical inference in spatial statistics
***********************************************************

Spatial statistics are valid if the variable of interest verifies `the assumption of second-order stationarity
<https://www.aspexit.com/en/fundamental-assumptions-of-the-variogram-second-order-stationarity-intrinsic-stationarity-what-is-this-all-about/>`_.
That is, if the three following assumptions are verified:

1. The mean of the variable of interest is stationary in space, i.e. constant over sufficiently large areas,
2. The variance of the variable of interest is stationary in space, i.e. constant over sufficiently large areas.
3. The covariance between two observations only depends on the spatial distance between them, i.e. no other factor than this distance plays a role in the spatial correlation of measurement errors.

A sufficiently large averaging area is an area expected to fit within the spatial domain studied.

In other words, for a reliable analysis, the DEM should:

1. Not contain systematic biases that do not average out over sufficiently large distances (e.g., shifts, tilts), but can contain pseudo-periodic biases (e.g., along-track undulations),
2. Not contain measurement errors that vary significantly in space.
3. Not contain factors that significantly affect the distribution of measurement errors, except for the spatial distance.

Quantifying the precision of a single DEM, or of a difference of DEMs
*********************************************************************

To statistically infer the precision of a DEM, the DEM has to be compared against independent elevation observations.

If the other elevation data is known to be of higher-precision, one can assume that the analysis of differences will represent the precision of the rougher DEM.
Otherwise, significant measurement errors can originate from both sets of elevation observations, and the analysis of differences will represent the mixed precision of the two.

TODO: complete with Hugonnet et al. (in prep)

Stable terrain: proxy for infering DEM precision
************************************************

When comparing elevation datasets, stable terrain is usually used a proxy

Metrics for DEM precision
*************************

The precision of DEMs has generally been reported as a single value indicating the random error at the scale of a single pixel, for example :math:`\pm 2` meters.

However, the significant variability of elevation measurement errors has been noted
In Hugonnet et al. (in prep),


Pixel-wise elevation measurement error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO


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


Workflow for DEM precision estimation
*************************************

Non-stationarity in elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Hugonnet et al. (in prep)

Multi-range spatial correlations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based Rolstad et al. (2009), Dehecq et al. (2020), Hugonnet et al. (in prep)

.. literalinclude:: code/spatialstats.py
        :lines: 26-27


Spatially integrated measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Rolstad et al. (2009), Hugonnet et al. (in prep)

Propagation of correlated errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Krige's relation (Webster & Oliver, 2007), Hugonnet et al. (in prep)

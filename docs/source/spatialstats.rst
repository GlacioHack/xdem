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

Spatial statistics for DEM precision estimation
-----------------------------------------------

Assumptions for statistical inference in spatial statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To statistically infer the precision of a DEM, the DEM has to be compared against independent elevation observations.

Significant measurement errors can originate from both sets of elevation observations, and the analysis of differences will represent the mixed precision of the two.
As there is no reason for a dependency between the elevation data sets, the analysis of elevation differences yields:

.. math::
        \sigma_{dh} = \sigma_{h_{\textrm{precision1}} - h_{\textrm{precision2}}} = \sqrt{\sigma_{h_{\textrm{precision1}}}^{2} + \sigma_{h_{\textrm{precision2}}}^{2}}

If the other elevation data is known to be of higher-precision, one can assume that the analysis of differences will represent only the precision of the rougher DEM.

.. math::
        \sigma_{dh} = \sigma_{h_{\textrm{higher precision}} - h_{\textrm{lower precision}}} \approx \sigma_{h_{\textrm{lower precision}}}


TODO: complete with Hugonnet et al. (in prep)

Using stable terrain as a proxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When comparing elevation datasets, stable terrain is usually used a proxy

Workflow for DEM precision estimation
-------------------------------------

Non-stationarity in elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. minigallery:: xdem.spatialstats.nd_binning
        :add-heading:

Quantify and model non-stationarites
""""""""""""""""""""""""""""""""""""

TODO: Add this section based on Hugonnet et al. (in prep)

.. literalinclude:: code/spatialstats.py
        :lines: 16-17

Standardize elevation differences for further analysis
""""""""""""""""""""""""""""""""""""""""""""""""""""""


Spatial correlation of elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based Rolstad et al. (2009), Dehecq et al. (2020), Hugonnet et al. (in prep)

Quantify and model spatial correlations
"""""""""""""""""""""""""""""""""""""""

.. literalinclude:: code/spatialstats.py
        :lines: 19-20

For a single range model:

.. literalinclude:: code/spatialstats.py
        :lines: 22-23

For multiple range model:

.. literalinclude:: code/spatialstats.py
        :lines: 25-26

.. plot:: code/spatialstats_plot_vgm.py

Spatially integrated measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Rolstad et al. (2009), Hugonnet et al. (in prep)

Propagation of correlated errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Krige's relation (Webster & Oliver, 2007), Hugonnet et al. (in prep)


Metrics for DEM precision
-------------------------

Historically, the precision of DEMs has been reported as a single value indicating the random error at the scale of a single pixel, for example :math:`\pm 2` meters.

However, there is several limitations to this metric:

- studies have shown significant variability of elevation measurement errors with terrain attributes, such as the slope, but also with the type of terrain


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

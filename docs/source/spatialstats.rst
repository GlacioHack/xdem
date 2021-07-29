.. _spatialstats:

Spatial statistics
==================

Spatial statistics, also referred to as `geostatistics <https://en.wikipedia.org/wiki/Geostatistics>`_, are essential
for the analysis of observations distributed in space.
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

Using stable terrain as a proxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stable terrain is the terrain that has supposedly not been subject to any elevation change. For example bare-rock, or
generally almost all terrain excluding glaciers, snow and forests.

Due to the sparsity of synchronous acquisitions, elevation data cannot be easily compared over similar periods. Thus,
when comparing elevation data, stable terrain is used a proxy to assess the precision of a DEM on all its terrain,
including moving terrain that is generally of greater interest for analysis.

As shown in Hugonnet et al. (in prep), accounting for :ref:`spatialstats_nonstationarity` is needed to reliably
use stable terrain as a proxy for other types of terrain.

Workflow for DEM precision estimation
-------------------------------------

.. _spatialstats_nonstationarity:

Non-stationarity in elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantify and model non-stationarites
""""""""""""""""""""""""""""""""""""

Non-stationarities in elevation measurement errors correspond to a variability of the precision in the elevation
observations with certain factors, that can be terrain- or instrument-related.
In statistical terms, it corresponds to an `heteroscedasticity <https://en.wikipedia.org/wiki/Heteroscedasticity>`_
of elevation observations.

TODO: Add equation

Owing to the large number of samples of elevation data, we can easily estimate this variability by `binning *
<https://en.wikipedia.org/wiki/Data_binning>`_ the data along explanatory variables:

.. literalinclude:: code/spatialstats.py
        :lines: 17-19

Most typically, the explanatory variables used are:
    - the terrain slope and terrain curvature (see :ref:`terrain_attributes) that can explain a large part of the
terrain-related variability in measurement error,
    - the quality of stereo-correlation that can explain a large part of the measurement error of DEMs generated by
stereophotogrammetry,
    - the interferometric coherence that can explain a large part of the measurement error of DEMs generated by
`InSAR <https://en.wikipedia.org/wiki/Interferometric_synthetic-aperture_radar>`_.

Standardize elevation differences for further analysis
""""""""""""""""""""""""""""""""""""""""""""""""""""""

TODO: Add text

TODO: Add equation

TODO: Add a new gallery example

.. minigallery:: xdem.spatialstats.nd_binning
        :add-heading: Example that deal with non-stationarities

.. _spatialstats_spatialcorr:

Spatial correlation of elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based Rolstad et al. (2009), Dehecq et al. (2020), Hugonnet et al. (in prep)

.. minigallery:: xdem.spatialstats.sample_multirange_variogram
        :add-heading: Examples that deal with spatial correlations


Quantify spatial correlations
"""""""""""""""""""""""""""""

Estimate empirical variogram:

.. literalinclude:: code/spatialstats.py
        :lines: 24-25

Model spatial correlations
""""""""""""""""""""""""""

Fit a multiple-range model:

.. literalinclude:: code/spatialstats.py
        :lines: 27-28

.. _spatialstats_errorpropag:

Spatially integrated measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deduce an effective sample size, and elevation measurement error:

.. literalinclude:: code/spatialstats.py
        :lines: 30-33

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

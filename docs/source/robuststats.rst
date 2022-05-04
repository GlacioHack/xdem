.. _robuststats:

Robust statistics
==================

Digital Elevation Models often contain outliers that hamper further analysis.
In order to mitigate their effect on DEM analysis, ``xdem`` integrates `robust statistics <https://en.wikipedia.org/wiki/Robust_statistics>`_
methods at different levels.
These methods can be used to robustly fit functions necessary to perform DEM alignment (see :ref:`coregistration`, :ref:`biascorr`), or to provide
robust statistical measures equivalent to the mean, the standard deviation or the covariance of a sample when analyzing DEM precision with
:ref:`spatialstats`.

Yet, there is a downside to robust statistical measures. Those can yield less precise estimates for small samples sizes and,
in some cases, hide patterns inherent to the data. This is why, when outliers exhibit idenfiable patterns, it is better
to first resort to outlier filtering (see :ref:`filters`) and perform analysis using traditional statistical measures.

.. contents:: Contents 
   :local:

.. _robuststats_meanstd:

Measures of central tendency and dispersion
-------------------------------------------

Central tendency
^^^^^^^^^^^^^^^^

The `central tendency <https://en.wikipedia.org/wiki/Central_tendency>`_ represents the central value of a sample, and is
core to the analysis of sample accuracy (see :ref:`intro`). It is most often measured by the `mean <https://en.wikipedia.org/wiki/Mean>`_.
However, the mean is a measure sensitive to outliers. Therefore, in many cases (e.g., when working with unfiltered
DEMs) using the `median <https://en.wikipedia.org/wiki/Median>`_ as measure of central tendency is preferred.

When working with weighted data, the `weighted median <https://en.wikipedia.org/wiki/Weighted_median>`_ which corresponds
to the 50\ :sup:`th` `weighted percentile <https://en.wikipedia.org/wiki/Percentile#Weighted_percentile>`_ can be
used as a robust measure of central tendency.

The median is used by default in the alignment routines of :ref:`coregistration` and :ref:`biascorr`.

Dispersion
^^^^^^^^^^

The `statistical dispersion <https://en.wikipedia.org/wiki/Statistical_dispersion>`_ represents the spread of a sample,
and is core to the analysis of sample precision (see :ref:`intro`). It is typically measured by the `standard deviation
<https://en.wikipedia.org/wiki/Standard_deviation>`_.
However, very much like the mean, the standard deviation is a measure sensitive to outliers. The median equivalent of a
standard deviation is the normalized median absolute deviation (NMAD), which corresponds to the `median absolute deviation
<https://en.wikipedia.org/wiki/Median_absolute_deviation>`_ scaled by a factor of ~1.4826 to match the dispersion of a
normal distribution. It has been shown to provide more robust measures of dispersion with outliers when working
with DEMs (e.g., `Höhle and Höhle (2009) <https://doi.org/10.1016/j.isprsjprs.2009.02.003>`_).
It is defined as:

.. math::
        \textrm{NMAD}(x) = 1.4826 \cdot \textrm{median}_{i} \left ( \mid x_{i} - \textrm{median}(x) \mid \right )

where :math:`x` is the sample.

The half difference between 84\ :sup:`th` and 16\ :sup:`th` percentiles, or the absolute 68\ :sup:`th` percentile
can also be used as a robust dispersion measure equivalent to the standard deviation.

.. code-block:: python

    nmad = xdem.spatialstats.nmad

When working with weighted data, the difference between the 84\ :sup:`th` and 16\ :sup:`th` `weighted percentile <https://en.wikipedia.org
/wiki/Percentile#Weighted_percentile>`_, or the absolute 68\ :sup:`th` weighted percentile can be used as a robust measure of dispersion.

The NMAD is used by default for estimating elevation measurement errors in :ref:`spatialstats`.

.. _robuststats_corr:

Measures of correlation
-----------------------

Correlation between samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `covariance <https://en.wikipedia.org/wiki/Covariance>`_ is the measure generally used to estimate the joint variability
of samples, often normalized to a `correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.
Again, the variance and covariance are sensitive measures to outliers. It is therefore preferable to compute such measures
by filtering the data, or using robust estimators.

TODO

Spatial auto-correlation of a sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Variogram <https://en.wikipedia.org/wiki/Variogram>`_ analysis exploits statistical measures equivalent to the covariance,
and is therefore also subject to outliers.
Based on `scikit-gstat <https://mmaelicke.github.io/scikit-gstat/index.html>`_, ``xdem`` allows to specify robust variogram
estimators such as Dowd's variogram based on medians (`Dowd (1984) <https://en.wikipedia.org/wiki/Variogram>`_) defined as:

.. math::
        2\gamma (h) = 2.198 \cdot \textrm{median}_{i} \left ( Z_{x_{i}} - Z_{x_{i+h}} \right )

where :math:`h` is the spatial lag and :math:`Z_{x_{i}}` is the value of the sample at the location :math:`x_{i}`.

Dowd's variogram is used by default to estimate spatial auto-correlation of elevation measurement errors in :ref:`spatialstats`.

.. _robuststats_regression:

Regression analysis
-------------------

Least-square loss functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When performing least-squares linear regression, the traditional `loss functions <https://en.wikipedia.org/wiki/Loss_
function>`_ that are used are not robust to outliers.

A robust soft L1 loss default is used by default when ``xdem`` performs least-squares regression through `scipy.optimize
<https://docs.scipy.org/doc/scipy/reference/optimize.html#>`_.

Robust estimators
^^^^^^^^^^^^^^^^^

Other estimators than ordinary least-squares can be used for linear estimations.
The :ref:`coregistration` and :ref:`biascorr` methods encapsulate some of those robust methods provided by `sklearn.linear_models
<https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors>`_:

- The Random sample consensus estimator `RANSAC <https://en.wikipedia.org/wiki/Random_sample_consensus>`_,
- The `Theil-Sen <https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator>`_ estimator,
- The `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ estimator.


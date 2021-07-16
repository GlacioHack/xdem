Robust statistics
==================

Digital Elevation Models often contain outliers that hamper further analysis.
In order to deal with outliers, ``xdem`` integrates `robust statistics <https://en.wikipedia.org/wiki/Robust_statistics>`_
methods at different levels.
For instance, those can be used robustly fit functions necessary to perform alignment (:ref:`coreg`, :ref:`biascorr`), or to provide
robust statistical measures equivalent to the mean, the standard deviation or the covariance of a sample when dealing with
:ref:`spatialstats`.

The downside of robust statistical measures is that those can yield less precise estimates for small samples sizes and,
in some cases, hide patterns inherent to the data by smoothing.
As a consequence, when outliers exhibit idenfiable patterns, it is better to first resort to outlier filtering (:ref:`filters`)
and perform analysis using traditional statistical measures.

.. contents:: Contents 
   :local:

Measures of central tendency and dispersion of a sample
--------------------------------------------------------

Central tendency
^^^^^^^^^^^^^^^^

The `central tendency <https://en.wikipedia.org/wiki/Central_tendency>`_ represents the central value of a sample,
typically described by measures such as the `mean <https://en.wikipedia.org/wiki/Mean>`, and is mostly useful during
analysis of sample accuracy (see :ref:`intro`).
However, the mean is a measure sensitive to outliers. In many cases, for example when working on unfiltered DEMs, using
the `median <https://en.wikipedia.org/wiki/Median>`_ is therefore preferable.

When working with weighted data, the `weighted median <https://en.wikipedia.org/wiki/Weighted_median>`_ which corresponds
to the 50\ :sup:`th` `weighted percentile <https://en.wikipedia.org/wiki/Percentile#Weighted_percentile>`_, can also be
used as a robust measure of central tendency.

The median is used by default alignment routines in :ref:`coreg` and :ref:`biascorr`.

Dispersion
^^^^^^^^^^

The `statistical dispersion <https://en.wikipedia.org/wiki/Statistical_dispersion>`_ represents the spread of a sample,
typically described by measures such as the `standard deviation <https://en.wikipedia.org/wiki/Standard_deviation>`_, and
is a useful metric in the analysis of sample precision (see :ref:`intro`).
However, the standard deviation is a measure sensitive to outliers. The normalized median absolute deviation (NMAD), which
corresponds to the `median absolute deviation <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_ scaled by a factor
of ~1.4826 to match the dispersion of a normal distribution, is the median equivalent of a standard deviation and has been shown to
provide more robust when working with DEMs (e.g., `Höhle and Höhle (2009) <https://doi.org/10.1016/j.isprsjprs.2009.02.003>`_).
The half difference between 84\ :sup:`th` and 16\ :sup:`th` percentiles, or the absolute 68\ :sup:`th` percentile
can also be used as a robust dispersion measure equivalent to the standard deviation.

.. code-block:: python
        nmad = xdem.spatial_tools.nmad(ddem.data)

When working with weighted data, the difference between the 84th and 16th `weighted percentile <https://en.wikipedia.org
/wiki/Percentile#Weighted_percentile>`_, or the absolute 68\ :sup:`th` weighted percentile can be used as a robust measure of dispersion.

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
estimators such as Dowd's variogram based on medians (`Dowd (1984) <https://en.wikipedia.org/wiki/Variogram>`_).

Regression analysis
-------------------

``xdem`` encapsulates methods from scipy and sklearn to perform robust regression for :ref:`coreg` and :ref:`biascorr`.

Robust loss functions
^^^^^^^^^^^^^^^^^^^^^

Based on `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html#>`_ and specific `loss functions
<https://en.wikipedia.org/wiki/Loss_function>`_, robust least-squares can be performed.

Robust estimators
^^^^^^^^^^^^^^^^^

Based on `sklearn.linear_models <https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outlier
s-and-modeling-errors>`_, robust estimator such as `RANSAC <https://en.wikipedia.org/wiki/Random_sample_consensus>`_,
`Theil-Sen <https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator>`_, or the `Huber loss function <https://en.wikipedia.org/wiki/Huber_loss>`_
are available for robust function fitting.


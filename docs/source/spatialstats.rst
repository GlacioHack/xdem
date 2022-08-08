.. _spatialstats:

Spatial statistics
==================

Spatial statistics, also referred to as `geostatistics <https://en.wikipedia.org/wiki/Geostatistics>`_, are essential
for the analysis of observations distributed in space.
To analyze DEMs, xDEM integrates spatial statistics tools specific to DEMs described in recent literature,
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

.. _spatialstats_intro:

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

.. plot:: code/spatialstats_stationarity_assumption.py
    :width: 90%

In other words, for a reliable analysis, the DEM should:

    1. Not contain systematic biases that do not average out over sufficiently large distances (e.g., shifts, tilts), but can contain pseudo-periodic biases (e.g., along-track undulations),
    2. Not contain measurement errors that vary significantly across space.
    3. Not contain factors that affect the spatial distribution of measurement errors, except for the distance between observations.

Quantifying the precision of a single DEM, or of a difference of DEMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To statistically infer the precision of a DEM, it is compared against independent elevation observations.

Significant measurement errors can originate from both sets of elevation observations, and the analysis of differences will represent the mixed precision of the two.
As there is no reason for a dependency between the elevation data sets, the analysis of elevation differences yields:

.. math::
        \sigma_{dh} = \sigma_{h_{\textrm{precision1}} - h_{\textrm{precision2}}} = \sqrt{\sigma_{h_{\textrm{precision1}}}^{2} + \sigma_{h_{\textrm{precision2}}}^{2}}

If the other elevation data is known to be of higher-precision, one can assume that the analysis of differences will represent only the precision of the rougher DEM.

.. math::
        \sigma_{dh} = \sigma_{h_{\textrm{higher precision}} - h_{\textrm{lower precision}}} \approx \sigma_{h_{\textrm{lower precision}}}

Using stable terrain as a proxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stable terrain is the terrain that has supposedly not been subject to any elevation change. It often refers to bare-rock,
and is generally computed by simply excluding glaciers, snow and forests.

Due to the sparsity of synchronous acquisitions, elevation data cannot be easily compared for simultaneous acquisition
times. Thus, stable terrain is used a proxy to assess the precision of a DEM on all its terrain,
including moving terrain that is generally of greater interest for analysis.

As shown in Hugonnet et al. (in prep), accounting for :ref:`spatialstats_nonstationarity` is needed to reliably
use stable terrain as a proxy for other types of terrain.

.. _spatialstats_metrics:

Metrics for DEM precision
-------------------------

Historically, the precision of DEMs has been reported as a single value indicating the random error at the scale of a
single pixel, for example :math:`\pm 2` meters at the 1\ :math:`\sigma` `confidence level <https://en.wikipedia.org/wiki/Confidence_interval>`_.

However, there is some limitations to this simple metric:

    - the variability of the pixel-wise precision is not reported. The pixel-wise precision can vary depending on terrain- or instrument-related factors, such as the terrain slope. In rare occurences, part of this variability has been accounted in recent DEM products, such as TanDEM-X global DEM that partitions the precision between flat and steep slopes (`Rizzoli et al. (2017) <https://doi.org/10.1016/j.isprsjprs.2017.08.008>`_),
    - the area-wise precision of a DEM is generally not reported. Depending on the inherent resolution of the DEM, and patterns of noise that might plague the observations, the precision of a DEM over a surface area can vary significantly.

Pixel-wise elevation measurement error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pixel-wise measurement error corresponds directly to the dispersion :math:`\sigma_{dh}` of the sample :math:`dh`.

To estimate the pixel-wise measurement error for elevation data, two issues arise:

    1. The dispersion :math:`\sigma_{dh}` cannot be estimated directly on changing terrain,
    2. The dispersion :math:`\sigma_{dh}` can show important non-stationarities.

The section :ref:`spatialstats_nonstationarity` describes how to quantify the measurement error as a function of
several explanatory variables by using stable terrain as a proxy.

Spatially-integrated elevation measurement error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `standard error <https://en.wikipedia.org/wiki/Standard_error>`_ of a statistic is the dispersion of the
distribution of this statistic. For spatially distributed samples, the standard error of the mean corresponds to the
error of a mean (or sum) of samples in space.

The standard error :math:`\sigma_{\overline{dh}}` of the mean :math:`\overline{dh}` of the elevation changes
samples :math:`dh` can be written as:

.. math::

        \sigma_{\overline{dh}} = \frac{\sigma_{dh}}{\sqrt{N}},

where :math:`\sigma_{dh}` is the dispersion of the samples, and :math:`N` is the number of **independent** observations.

To estimate the standard error of the mean for elevation data, two issue arises:

    1. The dispersion of elevation differences :math:`\sigma_{dh}` is not stationary, a necessary assumption for spatial statistics.
    2. The number of pixels in the DEM :math:`N` does not equal the number of independent observations in the DEMs, because of spatial correlations.

The sections :ref:`spatialstats_corr` and :ref:`spatialstats_errorpropag` describe how to account for spatial correlations
and use those to integrate and propagate measurement errors in space.

Workflow for DEM precision estimation
-------------------------------------

.. _spatialstats_nonstationarity:

Non-stationarity in elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elevation data contains significant non-stationarities in elevation measurement errors.

xDEM provides tools to **quantify** these non-stationarities along several explanatory variables,
**model** those numerically to estimate an elevation measurement error, and **standardize** them for further analysis.

Quantify and model non-stationarites
""""""""""""""""""""""""""""""""""""

Non-stationarities in elevation measurement errors correspond to a variability of the precision in the elevation
observations with certain explanatory variables that can be terrain- or instrument-related.
In statistical terms, it corresponds to an `heteroscedasticity <https://en.wikipedia.org/wiki/Heteroscedasticity>`_
of elevation observations.

.. math::
    \sigma_{dh} = \sigma_{dh}(\textrm{var}_{1},\textrm{var}_{2}, \textrm{...}) \neq \textrm{constant}

Owing to the large number of samples of elevation data, we can easily estimate this variability by `binning
<https://en.wikipedia.org/wiki/Data_binning>`_ the data and estimating the statistical dispersion (see
:ref:`robuststats_meanstd`) across several explanatory variables using :func:`xdem.spatialstats.nd_binning`.

.. literalinclude:: code/spatialstats.py
        :lines: 18-19
        :language: python

.. plot:: code/spatialstats_nonstationarity_slope.py
    :width: 90%

The most common explanatory variables are:

    - the terrain slope and terrain curvature (see :ref:`terrain_attributes`) that can explain a large part of the terrain-related variability in measurement error,
    - the quality of stereo-correlation that can explain a large part of the measurement error of DEMs generated by stereophotogrammetry,
    - the interferometric coherence that can explain a large part of the measurement error of DEMs generated by `InSAR <https://en.wikipedia.org/wiki/Interferometric_synthetic-aperture_radar>`_.

Once quantified, the non-stationarities can be modelled numerically by linear interpolation across several
variables using :func:`xdem.spatialstats.interp_nd_binning`.

.. literalinclude:: code/spatialstats.py
        :lines: 22
        :language: python

Standardize elevation differences for further analysis
""""""""""""""""""""""""""""""""""""""""""""""""""""""

In order to verify the assumptions of spatial statistics and be able to use stable terrain as a reliable proxy in
further analysis (see :ref:`spatialstats_intro`), `standardization <https://en.wikipedia.org/wiki/Standard_score>`_
of the elevation differences are required to reach a stationary variance.

.. plot:: code/spatialstats_standardizing.py
    :width: 90%

For application to DEM precision estimation, the mean is already centered on zero and the variance is non-stationary,
which yields:

.. math::
    z_{dh} = \frac{dh(\textrm{var}_{1}, \textrm{var}_{2}, \textrm{...})}{\sigma_{dh}(\textrm{var}_{1}, \textrm{var}_{2}, \textrm{...})}

where :math:`z_{dh}` is the standardized elevation difference sample.

Code-wise, standardization is as simple as a division of the elevation differences ``dh`` using the estimated measurement
error:

.. literalinclude:: code/spatialstats.py
        :lines: 25
        :language: python

To later de-standardize estimations of the dispersion of a given subsample of elevation differences,
possibly after further analysis of :ref:`spatialstats_corr` and :ref:`spatialstats_errorpropag`,
one simply needs to apply the opposite operation.

For a single pixel :math:`\textrm{P}`, the dispersion is directly the elevation measurement error evaluated for the
explanatory variable of this pixel as, per construction, :math:`\sigma_{z_{dh}} = 1`:

.. math::
    \sigma_{dh}(\textrm{P}) = 1 \cdot \sigma_{dh}(\textrm{var}_{1}(\textrm{P}), \textrm{var}_{2}(\textrm{P}), \textrm{...})

For a mean of pixels :math:`\overline{dh}\vert_{\mathbb{S}}` in the subsample :math:`\mathbb{S}`, the standard error of the mean
of the standardized data :math:`\overline{\sigma_{z_{dh}}}\vert_{\mathbb{S}}` can be de-standardized by multiplying by the
average measurement error of the pixels in the subsample, evaluated through the explanatory variables of each pixel:

.. math::
    \sigma_{\overline{dh}}\vert_{\mathbb{S}} = \sigma_{\overline{z_{dh}}}\vert_{\mathbb{S}} \cdot \overline{\sigma_{dh}(\textrm{var}_{1}, \textrm{var}_{2}, \textrm{...})}\vert_{\mathbb{S}}

Estimating the standard error of the mean of the standardized data :math:`\sigma_{\overline{z_{dh}}}\vert_{\mathbb{S}}`
requires an analysis of spatial correlation and a spatial integration of this correlation, described in the next sections.

.. minigallery:: xdem.spatialstats.nd_binning
        :add-heading: Examples that deal with non-stationarities
        :heading-level: "

.. _spatialstats_corr:

Spatial correlation of elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spatial correlation of elevation measurement errors correspond to a dependency between measurement errors of spatially
close pixels in elevation data. Those can be related to the resolution of the data (short-range correlation), or to
instrument noise and deformations (mid- to long-range correlations).

xDEM provides tools to **quantify** these spatial correlation with pairwise sampling optimized for grid data and to
**model** correlations simultaneously at multiple ranges.

Quantify spatial correlations
"""""""""""""""""""""""""""""

`Variograms <https://en.wikipedia.org/wiki/Variogram>`_ are functions that describe the spatial correlation of a sample.
The variogram :math:`2\gamma(h)` is a function of the distance between two points, referred to as spatial lag :math:`l`
(usually noted :math:`h`, here avoided to avoid confusion with the elevation and elevation differences).
The output of a variogram is the correlated variance of the sample.

.. math::
        2\gamma(l) = \textrm{var}\left(Z(\textrm{s}_{1}) - Z(\textrm{s}_{2})\right)

where :math:`Z(\textrm{s}_{i})` is the value taken by the sample at location :math:`\textrm{s}_{i}`, and sample positions
:math:`\textrm{s}_{1}` and :math:`\textrm{s}_{2}` are separated by a distance :math:`l`.

For elevation differences :math:`dh`, this translates into:

.. math::
        2\gamma_{dh}(l) = \textrm{var}\left(dh(\textrm{s}_{1}) - dh(\textrm{s}_{2})\right)

The variogram essentially describes the spatial covariance :math:`C` in relation to the variance of the entire sample
:math:`\sigma_{dh}^{2}`:

.. math::
        \gamma_{dh}(l) = \sigma_{dh}^{2} - C_{dh}(l)

.. plot:: code/spatialstats_variogram_covariance.py
    :width: 90%

Empirical variograms are variograms estimated directly by `binned <https://en.wikipedia.org/wiki/Data_binning>`_ analysis
of variance of the data. Historically, empirical variograms were estimated for point data by calculating all possible
pairwise differences in the samples. This amounts to :math:`N^2` pairwise calculations for :math:`N` samples, which is
not well-suited to grid data that contains many millions of points and would be impossible to comupute. Thus, in order
to estimate a variogram for large grid data, subsampling is necessary.

Random subsampling of the grid samples used is a solution, but often unsatisfactory as it creates a clustering
of pairwise samples that unevenly represents lag classes (most pairwise differences are found at mid distances, but too
few at short distances and long distances).

To remedy this issue, xDEM provides :func:`xdem.spatialstats.sample_empirical_variogram`, an empirical variogram estimation tool
that encapsulates a pairwise subsampling method described in ``skgstat.MetricSpace.RasterEquidistantMetricSpace``.
This method compares pairwise distances between a center subset and equidistant subsets iteratively across a grid, based on
`sparse matrices <https://en.wikipedia.org/wiki/Sparse_matrix>`_ routines computing pairwise distances of two separate
subsets, as in `scipy.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_
(instead of using pairwise distances within the same subset, as implemented in most spatial statistics packages).
The resulting pairwise differences are evenly distributed across the grid and across lag classes (in 2 dimensions, this
means that lag classes separated by a factor of :math:`\sqrt{2}` have an equal number of pairwise differences computed).

.. literalinclude:: code/spatialstats.py
        :lines: 28-29
        :language: python

The variogram is returned as a ``pd.Dataframe`` object.

With all spatial lags sampled evenly, estimating a variogram requires significantly less samples, increasing the
robustness of the spatial correlation estimation and decreasing computing time!

Model spatial correlations
""""""""""""""""""""""""""

Once an empirical variogram is estimated, fitting a function model allows to simplify later analysis by directly
providing a function form (e.g., for kriging equations, or uncertainty analysis - see :ref:`spatialstats_errorpropag`),
which would otherwise have to be numerically modelled.

Generally, in spatial statistics, a single model is used to describe the correlation in the data.
In elevation data, however, spatial correlations are observed at different scales, which requires fitting a sum of models at
multiple ranges (introduced in `Rolstad et al. (2009) <https://doi.org/10.3189/002214309789470950>`_ for glaciology
applications).

This can be performed through the function :func:`xdem.spatialstats.fit_sum_model_variogram`, which expects as input a
``pd.Dataframe`` variogram.

.. literalinclude:: code/spatialstats.py
        :lines: 31
        :language: python

.. minigallery:: xdem.spatialstats.sample_empirical_variogram
        :add-heading: Examples that deal with spatial correlations
        :heading-level: "

.. _spatialstats_errorpropag:

Spatially integrated measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After quantifying and modelling spatial correlations, those an effective sample size, and elevation measurement error:

.. literalinclude:: code/spatialstats.py
        :lines: 33

TODO: Add this section based on Rolstad et al. (2009), Hugonnet et al. (in prep)

Propagation of correlated errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Krige's relation (Webster & Oliver, 2007), Hugonnet et al. (in prep)



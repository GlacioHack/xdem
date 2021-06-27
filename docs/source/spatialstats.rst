Spatial statistics
==================

Spatial statistics, also referred to as geostatistics, are essential for the analysis of observations distributed in space.
To analyze DEMs, ``xdem`` integrates spatial statistics tools specific to DEMs based on recent literature, and with routines partly relying on `scikit-gstat <https://mmaelicke.github.io/scikit-gstat/index.html>`_.

The spatial statistics tools can be used to:
    - account for non-stationarities of elevation measurement errors (e.g., varying precision of DEMs with terrain slope),
    - quantify the spatial correlation in DEMs (e.g., native spatial resolution, instrument noise),
    - estimate robust errors for observations integrated in space (e.g., average or sum of samples),
    - propagate errors between spatial ensembles at different scales (e.g., sum of glacier volume changes).

More details below.

.. contents:: Contents 
   :local:


Introduction: why is it complex to assess DEM accuracy and precision?
*********************************************************************

Digital Elevation Models are a numerical representations of elevation. They are generated from different instruments (radiometer, radar, lidar), acquired in different conditions (ground, airborne, satellite), and using different post-processing techniques (stereophotogrammetry, interferometry, etc.).

While some complexities are specific to certain instruments, all DEMs generally have:
    - an **arbitrary Ground Sampling Distance (GSD)** that does not necessarily represent their underlying spatial resolution,
    - an **georeferenced positioning subject to shifts, tilts or other deformations** due to inherent instrument errors, noise, or associated post-processing schemes,
    - a **large number of outliers** that can originate from various sources (e.g., photogrammetric blunders, clouds).

DEM accuracy or DEM precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both DEM accuracy and precision can be of interest when analyzing DEMs:
    - the **accuracy** (systematic error) of a DEM describes how close a DEM is to the true location of measured elevations on the Earth's surface,
    - the **precision** (random error) of a DEM describes the typical spread of its error in measurement, independently of a possible bias from the true positioning.

Absolute or relative accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The measure of accuracy can be further split into two:
    - the **absolute accuracy** of a DEM is the average shift to the true positioning. Studies interested in analyzing features of a single DEM might give great importance to this potential bias, which can be easily removed through a DEM co-registration with accurate, georeferenced point elevation data such as ICESat and ICESat-2 (:ref:`coregistration`).
    - the **relative accuracy** of a DEM is the potential shifts, tilts, and deformations in relation to other elevation data, not necessarily with true absolute referencing. Studies interested in comparing several DEMs in between them can focus only on this accuracy relative to the DEMs, by performed co-registration in between the DEMs and correcting for possible biases (:ref:`coregistration`, TODO: ref bias corrections).

As the **absolute accuracy** can be easily corrected a posteriori with an reference elevation dataset, we here only focus on **relative accuracy**, i.e. the biases between to DEMs co-registered relative one to another.

Pixel-wise elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



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


Relative spatial accuracy of a DEM
**********************************


Non-stationarity in elevation measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO: Add this section based on Hugonnet et al. (in prep)


Multi-range spatial correlations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based Rolstad et al. (2009), Dehecq et al. (2020), Hugonnet et al. (in prep)

Spatially integrated measurement errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Rolstad et al. (2009), Hugonnet et al. (in prep)

Propagation of correlated errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Add this section based on Krige's relation (Webster & Oliver, 2007

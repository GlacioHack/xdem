(reporting-metrics)=

# Reporting error metrics

Historically, the precision of DEMs has been reported as a single value indicating the random error at the scale of a
single pixel, for example $\pm 2$ meters at the 1$\sigma$ [confidence level](https://en.wikipedia.org/wiki/Confidence_interval).

However, there are some limitations to this simple metric:

> - the variability of the pixel-wise precision is not reported. The pixel-wise precision can vary depending on terrain- or instrument-related factors, such as the terrain slope. In rare occurrences, part of this variability has been accounted in recent DEM products, such as TanDEM-X global DEM that partitions the precision between flat and steep slopes ([Rizzoli et al. (2017)](https://doi.org/10.1016/j.isprsjprs.2017.08.008)),
> - the area-wise precision of a DEM is generally not reported. Depending on the inherent resolution of the DEM, and patterns of noise that might plague the observations, the precision of a DEM over a surface area can vary significantly.

### Pixel-wise elevation measurement error

The pixel-wise measurement error corresponds directly to the dispersion $\sigma_{dh}$ of the sample $dh$.

To estimate the pixel-wise measurement error for elevation data, two issues arise:

> 1. The dispersion $\sigma_{dh}$ cannot be estimated directly on changing terrain,
> 2. The dispersion $\sigma_{dh}$ can show important non-stationarities.

The section {ref}`spatialstats-heterosc` describes how to quantify the measurement error as a function of
several explanatory variables by using stable terrain as a proxy.

### Spatially-integrated elevation measurement error

The [standard error](https://en.wikipedia.org/wiki/Standard_error) of a statistic is the dispersion of the
distribution of this statistic. For spatially distributed samples, the standard error of the mean corresponds to the
error of a mean (or sum) of samples in space.

The standard error $\sigma_{\overline{dh}}$ of the mean $\overline{dh}$ of the elevation changes
samples $dh$ can be written as:

$$
\sigma_{\overline{dh}} = \frac{\sigma_{dh}}{\sqrt{N}},
$$

where $\sigma_{dh}$ is the dispersion of the samples, and $N$ is the number of **independent** observations.

To estimate the standard error of the mean for elevation data, two issue arises:

> 1. The dispersion of elevation differences $\sigma_{dh}$ is not stationary, a necessary assumption for spatial statistics.
> 2. The number of pixels in the DEM $N$ does not equal the number of independent observations in the DEMs, because of spatial correlations.

The sections {ref}`spatialstats-corr` and {ref}`spatialstats-errorpropag` describe how to account for spatial correlations
and use those to integrate and propagate measurement errors in space.

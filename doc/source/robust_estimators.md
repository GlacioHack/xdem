(robust-estimators)=

# Need for robust estimators

Elevation data often contain outliers that can be traced back to instrument acquisition or processing artefacts, and which hamper further analysis.

In order to mitigate their effect, the analysis of elevation data can integrate [robust statistics](https://en.wikipedia.org/wiki/Robust_statistics) at different levels:
- **Robust estimators for the central tendency and statistical dispersion** used during {ref}`coregistration`, {ref}`biascorr` and {ref}`uncertainty`,
- **Robust estimators for estimating spatial autocorrelation** applied to error propagation in {ref}`uncertainty`,
- **Robust optimizers for the fitting of parametric models** during {ref}`coregistration` and {ref}`biascorr`.

Yet, there is a downside to robust statistical estimators. Those can yield less precise estimates for small samples sizes and,
in some cases, hide patterns inherent to the data. This is why, when outliers show identifiable patterns, it can be better
to first resort to outlier filtering and perform analysis using traditional statistical measures.

```{important}
In xDEM, robust estimators are used everywhere by default.
```

(robuststats-meanstd)=

## Measures of central tendency and dispersion

### Central tendency

The [central tendency](https://en.wikipedia.org/wiki/Central_tendency) represents the central value of a sample, and is
core to the analysis of sample accuracy (see {ref}`accuracy-precision`). It is most often measured by the [mean](https://en.wikipedia.org/wiki/Mean).
However, the mean is a measure sensitive to outliers. Therefore, in many cases (e.g., when working with unfiltered
DEMs) using the [median](https://en.wikipedia.org/wiki/Median) as measure of central tendency is preferred.

When working with weighted data, the [weighted median](https://en.wikipedia.org/wiki/Weighted_median) which corresponds
to the 50{sup}`th` [weighted percentile](https://en.wikipedia.org/wiki/Percentile#Weighted_percentile) can be
used as a robust measure of central tendency.

The {func}`numpy.median` is used by default in the alignment routines of **{ref}`coregistration` and {ref}`biascorr`**.

```{eval-rst}
.. plot:: code/robust_mean_std.py
    :width: 90%
```

(robuststats-nmad)=

### Dispersion

The [statistical dispersion](https://en.wikipedia.org/wiki/Statistical_dispersion) represents the spread of a sample,
and is core to the analysis of sample precision (see {ref}`accuracy-precision`). It is typically measured by the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).
However, very much like the mean, the standard deviation is a measure sensitive to outliers.

The median equivalent of a standard deviation is the normalized median absolute deviation (NMAD), which corresponds to the [median absolute deviation](https://en.wikipedia.org/wiki/Median_absolute_deviation) scaled by a factor of ~1.4826 to match the dispersion of a
normal distribution. It is a more robust measure of dispersion with outliers, defined as:

$$
\textrm{NMAD}(x) = 1.4826 \cdot \textrm{median}_{i} \left ( \mid x_{i} - \textrm{median}(x) \mid \right )
$$

where $x$ is the sample.

```{note}
The NMAD estimator has a good synergy with {ref}`Dowd's variogram<robuststats-corr>` for spatial autocorrelation, as their median-based measure of dispersion is the same.
```

The half difference between 84{sup}`th` and 16{sup}`th` percentiles, or the absolute 68{sup}`th` percentile
can also be used as a robust dispersion measure equivalent to the standard deviation.
When working with weighted data, the difference between the 84{sup}`th` and 16{sup}`th` [weighted percentile](https://en.wikipedia.org/wiki/Percentile#Weighted_percentile), or the absolute 68{sup}`th` weighted percentile can be used as a robust measure of dispersion.

The {func}`gu.stats.nmad` is used by default in **{ref}`coregistration`, {ref}`biascorr` and {ref}`uncertainty`**.

(robuststats-corr)=

## Measures of spatial autocorrelation

[Variogram](https://en.wikipedia.org/wiki/Variogram) analysis exploits statistical measures equivalent to the covariance,
and is therefore also subject to outliers.
Based on [SciKit-GStat](https://mmaelicke.github.io/scikit-gstat/index.html), xDEM allows to specify robust variogram
estimators such as Dowd's variogram based on medians defined as:

$$
2\gamma (h) = 2.198 \cdot \textrm{median}_{i} \left ( Z_{x_{i}} - Z_{x_{i+h}} \right )
$$

where $h$ is the spatial lag and $Z_{x_{i}}$ is the value of the sample at the location $x_{i}$.

```{note}
Dowd's estimator has a good synergy with the {ref}`NMAD<robuststats-nmad>` for estimating the dispersion of the full sample, as their median-based measure of dispersion is the same (2.198 is the square of 1.4826).
```

Other estimators can be chosen from [SciKit-GStat's list of estimators](https://scikit-gstat.readthedocs.io/en/latest/reference/estimator.html).

Dowd's variogram is used by default to estimate spatial auto-correlation of elevation measurement errors in **{ref}`uncertainty`**.

```{eval-rst}
.. plot:: code/robust_vario.py
    :width: 90%
```

(robuststats-regression)=

## Regression analysis

### Least-square loss functions

When performing least-squares linear regression, the traditional [loss functions](https://en.wikipedia.org/wiki/Loss_function) that are used are not robust to outliers.

A robust soft L1 loss default is used by default to perform least-squares regression through [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html#) in **{ref}`coregistration` and {ref}`biascorr`**.

### Robust estimators

Other estimators than ordinary least-squares can be used for linear estimations.
The {ref}`coregistration` and {ref}`biascorr` methods encapsulate some of those robust methods provided by [sklearn.linear_models](https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors):

- The Random sample consensus estimator [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus),
- The [Theil-Sen](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator) estimator,
- The [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) estimator.

----------------

:::{admonition} References and more reading
:class: tip

**References:**
- [Dowd (1984)](https://doi.org/10.1007/978-94-009-3699-7_6), The Variogram and Kriging: Robust and Resistant Estimators,
- [Höhle and Höhle (2009)](https://doi.org/10.1016/j.isprsjprs.2009.02.003), Accuracy assessment of digital elevation models by means of robust statistical methods.
:::

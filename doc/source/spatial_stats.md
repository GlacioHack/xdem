---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: xdem-env
  language: python
  name: xdem
---
(spatial-stats)=

# Spatial statistics for error analysis

Performing error (or uncertainty) analysis of spatial variable, such as elevation data, requires **joint knowledge from
two scientific fields: spatial statistics and uncertainty quantification.**

Spatial statistics, also referred to as [geostatistics](https://en.wikipedia.org/wiki/Geostatistics) in geoscience,
is a body of theory for the analysis of spatial variables. It primarily relies on modelling the dependency of
variables in space (spatial autocorrelation) to better describe their spatial characteristics, and
utilize this in further quantitative analysis.

[Uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification) is the science of characterizing
uncertainties quantitatively, and includes a wide range of methods including in particular theoretical error propagation.
In measurement science, such as remote sensing, such uncertainty propagation is tightly linked with the field
of [metrology](https://en.wikipedia.org/wiki/Metrology).

In the following, we describe the basics assumptions and concepts required to perform a spatial uncertainty analysis of
elevation data, described in the **feature page {ref}`uncertainty`**.

## Assumptions for inference in spatial statistics

In spatial statistics, the covariance of a variable of interest is generally simplified into a spatial variogram, which
**describes the covariance only as function of the spatial lag** (spatial distance between two variable values).
However, to utilize this simplification of the covariance in subsequent analysis, the variable of interest must
respect [the assumption of second-order stationarity](https://www.aspexit.com/en/fundamental-assumptions-of-the-variogram-second-order-stationarity-intrinsic-stationarity-what-is-this-all-about/).
That is, verify the three following assumptions:

> 1. The mean of the variable of interest is stationary in space, i.e. constant over sufficiently large areas,
> 2. The variance of the variable of interest is stationary in space, i.e. constant over sufficiently large areas.
> 3. The covariance between two observations only depends on the spatial distance between them, i.e. no other factor than this distance plays a role in the spatial correlation of measurement errors.

```{eval-rst}
.. plot:: code/spatialstats_stationarity_assumption.py
    :width: 90%
```

In other words, for a reliable analysis, elevation data should:

> 1. Not contain elevation biases that do not average out over sufficiently large distances (e.g., shifts, tilts), but can contain pseudo-periodic biases (e.g., along-track undulations),
> 2. Not contain random elevation errors that vary significantly across space.
> 3. Not contain factors that affect the spatial distribution of elevation errors, except for the distance between observations.

While assumption **1.** is verified after coregistration and bias corrections, other assumptions are generally not
(e.g., larger errors on steep slope). To address this, we must estimate the variability of our random errors
(or heteroscedasticity), to then transform our data to achieve second-order stationarity.

```{note}
If there is no significant spatial variability in random errors in your elevation data (e.g., lidar),
you can **jump directly to the {ref}`spatialstats-corr` section**.
```

## Heteroscedasticity

Elevation [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity) corresponds to a variability in
precision of elevation observations, that are linked to terrain or instrument variables.

$$
\sigma_{dh} = \sigma_{dh}(\textrm{var}_{1},\textrm{var}_{2}, \textrm{...}) \neq \textrm{constant}
$$

While a single elevation difference (for a pixel or footpring) does not allow to capture random errors, larger samples
do. [Data binning](https://en.wikipedia.org/wiki/Data_binning), for instance, is a method that allows to estimate the
statistical spread of a sample per category, and can easily be used with one or more explanatory variables,
such as slope:

```{eval-rst}
.. plot:: code/spatialstats_heterosc_slope.py
    :width: 90%
```

Then, a model (parametric or not) can be fit to infer the variability of random errors at any data location.

## Standardization

In order to verify the assumptions of spatial statistics and be able to use stable terrain as a reliable proxy in
further analysis (see **{ref}`static-surfaces` guide page**), [standardization](https://en.wikipedia.org/wiki/Standard_score)
of the elevation differences by their mean $\mu$ and spread $\sigma$ are required to reach a stationary variance.

```{eval-rst}
.. plot:: code/spatialstats_standardizing.py
    :width: 90%
```

For elevation differences, the mean is already centered on zero but the variance is non-stationary,
which yields more simply:

$$
z_{dh} = \frac{dh(\textrm{var}_{1}, \textrm{var}_{2}, \textrm{...})}{\sigma_{dh}(\textrm{var}_{1}, \textrm{var}_{2}, \textrm{...})}
$$

where $z_{dh}$ is the standardized elevation difference sample.

(spatialstats-corr)=

## Spatial correlation of errors

Spatial correlation of elevation errors correspond to a dependency between measurement errors of spatially
close pixels in elevation data. Those can be related to the resolution of the data (short-range correlation), or to
instrument noise and deformations (mid- to long-range correlations).

[Variograms](https://en.wikipedia.org/wiki/Variogram) are functions that describe the spatial correlation of a sample.
The variogram $2\gamma(h)$ is a function of the distance between two points, referred to as spatial lag $d$.
The output of a variogram is the correlated variance of the sample.

$$
2\gamma(d) = \textrm{var}\left(Z(\textrm{s}_{1}) - Z(\textrm{s}_{2})\right)
$$

where $Z(\textrm{s}_{i})$ is the value taken by the sample at location $\textrm{s}_{i}$, and sample positions
$\textrm{s}_{1}$ and $\textrm{s}_{2}$ are separated by a distance $d$.

```{eval-rst}
.. plot:: code/spatialstats_variogram_covariance.py
    :width: 90%
```

For elevation differences $dh$, this translates into:

$$
2\gamma_{dh}(d) = \textrm{var}\left(dh(\textrm{s}_{1}) - dh(\textrm{s}_{2})\right)
$$

The variogram essentially describes the spatial covariance $C$ in relation to the variance of the entire sample
$\sigma_{dh}^{2}$:

$$
\gamma_{dh}(d) = \sigma_{dh}^{2} - C_{dh}(d)
$$


Variograms are estimated empirically by [data binning](https://en.wikipedia.org/wiki/Data_binning) depending on
the pairwise distance among data points, then modelled by functional forms called variogram models (e.g., spherical,
gaussian).

As pairwise combinations grow exponentially with data points, variograms are estimated using a random subsample of the
elevation data (whether gridded or point). Additionally, as elevation data usually contains patterns of correlation
at both short-range (close to pixel size) and long-range (close to DEM extent), the default pairwise sampling can be
tuned to perform well at all desired ranges.

```{note}
In xDEM, variograms use by default a subsample size of 1000 data points (leading to 500,000 pairwise differences estimated),
for computational efficiency.
xDEM also uses by default a random sampling method that selects a uniform amount of data points by pairwise log-distance,
in order to capture potential correlations at all ranges.
```

For more details on variography, **we refer to [the documentation of SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/userguide/userguide.html).**

## Error propagation

Once the heteroscedasticity $\sigma_{dh}(\textrm{var}_{1},\textrm{var}_{2}, \textrm{...})$ and spatial
correlation of errors $\gamma_{dh}(d)$ are both estimated, those can be used to propagate errors to derivatives
of the elevation data.

**For simple derivatives such as spatial derivatives** (e.g., mean or sum in an area), exact
[theoretical propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty) relying on the above components
can be employed.

For instance, to propagate errors to the mean of elevation differences in an area $A$ containing $N$ data points:

$$
\sigma_{\overline{dh}} = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} (1 - \gamma_{z_{dh}}(d_{ij})) \sigma_{dh_{i}} \sigma_{dh_{j}}
$$


where $d_{ij}$ is the distance between data point $i$ and $j$.

```{note}
xDEM implement several methods to approximate the above equation which scales exponentially with data points, to preserve
computational efficiency.
```

**For more complex derivatives such as terrain attributes**, the heteroscedasticity and spatial correlation of errors
can be used to constrain simulation methods that numerically generate realizations of the structure of errors.

For instance, to propagate errors to terrain slope, one can derive many realizations of random fields based on the error structure estimated
for the DEM. Then, for each realization, add the random field to the DEM, and derive its slope. Finally,
the error in slope can be estimated from the spread of all slope realizations.

----------------

:::{admonition} References and more reading
:class: tip

For random field generation, see for example [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/).

**References:**

- [Heuvelink et al. (1989)](https://doi.org/10.1080/02693798908941518), Propagation of errors in spatial modelling with GIS,
- [Rolstad et al. (2009)](http://dx.doi.org/10.3189/002214309789470950), Spatially integrated geodetic glacier mass balance and its uncertainty based on geostatistical analysis: Application to the western Svartisen ice cap, Norway,
- [Hugonnet et al. (2022)](https://doi.org/10.1109/jstars.2022.3188922), Uncertainty analysis of digital elevation models by spatial inference from stable terrain.

:::

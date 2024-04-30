(accuracy-precision)=

# Grasping accuracy and precision

Below, a small guide explaining what accuracy and precision are, and their relation to elevation data (or any spatial data!).

## Why do we need to understand accuracy and precision?

Elevation data comes from a wide range of instruments (optical, radar, lidar) acquiring in different conditions (ground, 
airborne, spaceborne) and relying on specific post-processing techniques (e.g., photogrammetry, interferometry).

While some complexities are specific to certain instruments and methods, all elevation data generally possesses:

- a [ground sampling distance](https://en.wikipedia.org/wiki/Ground_sample_distance), or pixel size, **that does not necessarily represent the underlying spatial resolution of the observations**,
- a [georeferencing](https://en.wikipedia.org/wiki/Georeferencing) **that can be subject to shifts, tilts or other deformations** due to inherent instrument errors, noise, or associated processing schemes,
- a large number of [outliers](https://en.wikipedia.org/wiki/Outlier) **that remain difficult to filter** as they can originate from various sources (e.g., blunders, clouds).

All of these factors lead to difficulties in assessing the reliability of elevation data, required to 
perform additional quantitative analysis, which calls for defining the concepts relating to accuracy and precision for elevation data.

## Accuracy and precision of elevation data

### What are accuracy and precision?

[Accuracy and precision](https://en.wikipedia.org/wiki/Accuracy_and_precision) describe random and systematic errors, respectively. 
A more accurate measurement is on average closer to the true value (less systematic error), while a more precise measurement has 
less spread in measurement error (less random error), as shown in the simple schematic below.

*Note: sometimes "accuracy" is also used to describe both types of errors, while "trueness" refers to systematic errors, as defined
in* [ISO 5725-1](https://www.iso.org/obp/ui/#iso:std:iso:5725:-1:ed-1:v1:en) *. Here, we use accuracy for systematic
errors as, to our knowledge, it is a more commonly used terminology for remote sensing applications.*

:::{figure} imgs/precision_accuracy.png
:width: 80%

Source: [antarcticglaciers.org](http://www.antarcticglaciers.org/glacial-geology/dating-glacial-sediments2/precision-and-accuracy-glacial-geology/), accessed 29.06.21.
:::

### Translating these concepts for elevation data

However, elevation data rarely consists of a single independent measurement but of a **series of measurement** (image grid, 
ground track) **related to a spatial support** (horizontal georeferencing, independent of height), which complexifies 
the notion of accuracy and precision. 

Due to this, spatially consistent systematic errors can arise in elevation data independently of the error in elevation itself, 
such as **affine biases** (systematic georeferencing shifts), in addition to **specific biases** known to exist locally 
(e.g., signal penetration in land cover type).

For random errors, a variability in error magnitude or **heteroscedasticity** is common in elevation data (e.g., 
large errors on steep slopes). Finally, spatially structured yet random patterns of errors (e.g., along-track undulations) 
also exist and force us to consider the **spatial correlation of errors**.

Translating the accuracy and precision concepts to elevation data, we can thus define:

- **Elevation accuracy** (systematic error) describes how close an elevation data is to the true elevation on the Earth's surface, both for errors **common to the entire spatial support**
(DEM grid, altimetric track) and errors **specific to a location** (pixel, footprint),
- **Elevation precision** (random error) describes the random spread of elevation error in measurement, independently of a possible bias from the true positioning, both for errors **structured over the spatial support** and **specific to a location**.

These categories are depicted in the figure below.

:::{figure} https://github.com/rhugonnet/dem_error_study/blob/main/figures/fig_2.png?raw=true
:width: 100%

Source: [Hugonnet et al. (2022)](https://doi.org/10.1109/jstars.2022.3188922).
:::

### Absolute or relative elevation accuracy

Accuracy is generally considered from two focus points:

- **Absolute elevation accuracy** describes systematic errors to the true positioning, usually important when analysis focuses on the exact location of topographic features at a specific epoch.
- **Relative elevation accuracy** describes systematic errors with reference to other elevation data that does not necessarily matches the true positioning, important for analyses interested in topographic change over time. 

## How to get the best accuracy and precision of your elevation data? 

### Quantifying and improving absolute and relative elevation accuracy

Misalignments due to poor absolute or relative accuracy are common in elevation datasets, and are usually assessed and 
corrected by performing **three-dimensional elevation coregistration and bias corrections to an independent source 
of elevation data**.

In the case of absolute accuracy, this independent source must be precise and accurate, such as altimetric data from
[ICESat](https://icesat.gsfc.nasa.gov/icesat/) and [ICESat-2](https://icesat-2.gsfc.nasa.gov/) elevations, or coarser yet 
quality-controlled DEMs themselves aligned on altimetric data such as the 
[Copernicus DEM](https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3).

To use coregistration and bias correction pipelines in xDEM, see the **feature pages on {ref}`coregistration` and {ref}`biascorr`**.

```{eval-rst}
.. minigallery:: xdem.coreg.Coreg
    :add-heading: Examples that use coregistration and bias corrections
```

### Quantifying and improving assessing elevation precision

While assessing accuracy is fairly straightforward as it consists of computing the mean differences (biases) between
two or multiple datasets, assessing precision of elevation data can be much more complex. The spread in measurement 
errors cannot be quantified by a single difference and require statistical inference. 

Assessing precision usually means applying **spatial statistics combined to uncertainty quantification**, 
to account for the spatial variability and the spatial correlation in errors. For this it is usually necessary, as 
for coregistration, to **rely on an independent source of elevation data on static surfaces similarly**. More background 
on this topic is available on the **{ref}`spatial-stats` guide page**.

To use spatial statistics for quantifying precision in xDEM, see **the feature page on {ref}`uncertainty`**.

Additionally, improving the precision of elevation data is sometimes possible by correcting random structured 
errors using, as for accuracy, **bias correction methods but here applied to pseudo-periodic errors**.

% Functions that are used in several examples create duplicate examples instead of being merged into the list.
% Circumventing manually by selecting functions used only once in each example for now.
```{eval-rst}
.. minigallery:: xdem.spatialstats.infer_heteroscedasticity_from_stable xdem.spatialstats.get_variogram_model_func xdem.spatialstats.sample_empirical_variogram
    :add-heading: Examples that use spatial statistics functions
```
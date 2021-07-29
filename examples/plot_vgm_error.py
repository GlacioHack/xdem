"""
Spatial correlation of elevation measurement errors
===================================================

Digital elevation models have elevation measurement errors that can vary with terrain or instrument-related variables
(see :ref:`sphx_glr_auto_examples_plot_nonstationary_error.py`), but those measurement errors are also often
`correlated in space <https://en.wikipedia.org/wiki/Spatial_analysis#Spatial_auto-correlation>`_.
While many DEM studies have been using short-range `variogram <https://en.wikipedia.org/wiki/Variogram>`_ to
estimate the correlation of elevation measurement errors (e.g., `Howat et al. (2008) <https://doi.org/10.1029/2008GL034496>`_
, `Wang and Kääb (2015) <https://doi.org/10.3390/rs70810117>`_), recent studies show that variograms of multiple ranges
provide larger, more reliable estimates of spatial correlation for DEMs (e.g., `Dehecq et al. (2020) <https://doi.org/10.3389/feart.2020.566802>`_
, `Hugonnet et al. (2021) <https://doi.org/10.1038/s41586-021-03436-z>`_).

Quantifying the spatial correlation in elevation measurement errors is essential to integrate measurement errors over
an area of interest (e.g, to estimate the error of a mean or sum of samples). Once the spatial correlations are quantified,
several methods exist to derive the related measurement error integrated in space (`Rolstad et al. (2009) <https://doi.org/10.3189/002214309789470950>`_
, Hugonnet et al. (in prep)). More details are available in :ref:`spatialstats`.

Here, we show an example in which we estimate spatially integrated elevation measurement errors for a DEM difference of
Longyearbyen glacier, demonstrated in :ref:`sphx_glr_auto_examples_plot_nuth_kaab.py`. We first quantify the spatial
correlations using :func:`xdem.spatialstats.sample_multirange_variogram` based on routines of `scikit-gstat
<https://mmaelicke.github.io/scikit-gstat/index.html>`_. We then model the empirical variogram using a sum of variogram
models using :func:`xdem.spatialstats.fit_sum_variogram`.
Finally, we integrate the variogram models for varying surface areas to estimate the spatially integrated elevation
measurement errors using :func:`xdem.spatialstats.neff_circ`, and empirically validate the improved robustness of
our results using :func:`xdem.spatialstats.patches_method`, an intensive Monte-Carlo sampling approach.

"""
# sphinx_gallery_thumbnail_number = 6
import matplotlib.pyplot as plt
import numpy as np
import xdem
import geoutils as gu

# %%
# We start by loading example files including a difference of DEMs at Longyearbyen glacier and the outlines to rasterize
# a glacier mask.
# Prior to differencing, the DEMs were aligned using :ref:`coregistration_nuthkaab` as shown in
# the :ref:`sphx_glr_auto_examples_plot_nuth_kaab.py` example. We later refer to those elevation differences as *dh*.

dh = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask_glacier = glacier_outlines.create_mask(dh)

# %%
# We remove values on glacier terrain in order to isolate stable terrain, our proxy for elevation measurement errors.
dh.data[mask_glacier] = np.nan

# %%
# We estimate the average per-pixel elevation measurement error on stable terrain, using both the standard deviation
# and normalized median absolute deviation. For this example, we do not account for the non-stationarity in elevation
# measurement errors quantified in :ref:`sphx_glr_auto_examples_plot_nonstationary_error.py`.
print('STD: {:.2f} meters.'.format(np.nanstd(dh.data)))
print('NMAD: {:.2f} meters.'.format(xdem.spatialstats.nmad(dh.data)))

# %%
# The two measures of dispersion are quite similar showing that, on average, there is a small influence of outliers on the
# elevation differences. The per-pixel precision is about :math:`\pm` 2.5 meters.
# **Does this mean that every pixel has an independent measurement error of** :math:`\pm` **2.5 meters?**
# Let's plot the elevation differences to visually check the quality of the data.
plt.figure(figsize=(8, 5))
_ = dh.show(ax=plt.gca(), cmap='RdYlBu', vmin=-4, vmax=4, cb_title='Elevation differences (m)')

# %%
# We clearly see that the residual elevation differences on stable terrain are not random. The positive and negative
# differences (blue and red, respectively) appear correlated over large distances. These correlated errors are what
# we aim to quantify.

# %%
# Additionally, we notice that the elevation differences are still polluted by unrealistically large elevation
# differences near glaciers, probably because the glacier inventory is more recent than the data, and the outlines are too small.
# To remedy this, we filter large elevation differences outside 4 NMAD.
dh.data[np.abs(dh.data) > 4 * xdem.spatialstats.nmad(dh.data)] = np.nan

# %%
# We plot the elevation differences after filtering to check that we successively removed the reminaing glacier signals.
plt.figure(figsize=(8, 5))
_ = dh.show(ax=plt.gca(), cmap='RdYlBu', vmin=-4, vmax=4, cb_title='Elevation differences (m)')

# %%
# To quantify the spatial correlation of the data, we sample an empirical variogram.
# The empirical variogram describes the variance between the elevation differences of pairs of pixels depending on their
# distance. This distance between pairs of pixels if referred to as spatial lag.
#
# To perform this procedure effectively, we use improved methods that provide efficient pairwise sampling methods for
# large grid data in `scikit-gstat <https://mmaelicke.github.io/scikit-gstat/index.html>`_, which are encapsulated
# conveniently by :func:`xdem.spatialstats.sample_multirange_variogram`:
df = xdem.spatialstats.sample_multirange_variogram(
    values=dh.data, gsd=dh.res[0], subsample=50, runs=30, n_variograms=10, estimator='cressie', random_state=42)

# %%
# We plot the empirical variogram:
xdem.spatialstats.plot_vgm(df)

# %%
# With this plot, it is hard to conclude anything! Properly visualizing the empirical variogram is one of the most
# important step. With grid data, we expect short-range correlations close to the resolution of the grid (~20-200
# meters), but also possibly longer range correlation due to instrument noise or alignment issues (~1-50 km) (Hugonnet et al., in prep).
#
# To better visualize the variogram, we can either change the axis to log-scale, but this might make it more difficult
# to later compare to variogram models. # Another solution is to split the variogram plot into subpanels, each with
# its own linear scale. Both are shown below.

# %%
# **Log scale:**
xdem.spatialstats.plot_vgm(df, xscale='log')

# %%
# **Subpanels with linear scale:**
xdem.spatialstats.plot_vgm(df, xscale_range_split=[100, 1000, 10000])

# %%
# We identify:
#   - a short-range (i.e., correlation length) correlation, likely due to effects of resolution. It has a large partial sill (correlated variance), meaning that the elevation measurement errors are strongly correlated until a range of ~100 m.
#   - a longer range correlation, with a smaller partial sill, meaning the part of the elevation measurement errors remain correlated over a longer distance.
#
# In order to show the difference between accounting only for the most noticeable, short-range correlation, or adding the
# long-range correlation, we fit this empirical variogram with two different models: a single spherical model, and
# the sum of two spherical models (two ranges). For this, we use :func:`xdem.spatialstats.fit_sum_variogram`, which
# is based on `scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_:
fun, params1 = xdem.spatialstats.fit_sum_variogram(['Sph'], empirical_variogram=df)

fun2, params2 = xdem.spatialstats.fit_sum_variogram(['Sph', 'Sph'], empirical_variogram=df)

xdem.spatialstats.plot_vgm(df,list_fit_fun=[fun, fun2],list_fit_fun_label=['Single-range model', 'Double-range model'],
                           xscale_range_split=[100, 1000, 10000])

# %%
# The sum of two spherical models fits better, accouting for the small partial sill at longer ranges. Yet this longer
# range partial sill (correlated variance) is quite small...
#
# **So one could ask himself: is it really important to account for this small additional "bump" in the variogram?**
#
# To answer this, we compute the precision of the DEM integrated over a certain surface area based on spatial integration of the
# variogram models using :func:`xdem.spatialstats.neff_circ`, with areas varying from pixel size to grid size.
# Numerical and exact integration of variogram is fast, allowing us to estimate errors for a wide range of areas radidly.

areas = np.linspace(20**2, 10000**2, 1000)

list_stderr_singlerange, list_stderr_doublerange, list_stderr_empirical = ([] for i in range(3))
for area in areas:

    # Number of effective samples integrated over the area for a single-range model
    neff_singlerange = xdem.spatialstats.neff_circ(area, [(params1[0], 'Sph', params1[1])])

    # For a double-range model
    neff_doublerange = xdem.spatialstats.neff_circ(area, [(params2[0], 'Sph', params2[1]),
                                                   (params2[2], 'Sph', params2[3])])

    # Convert into a standard error
    stderr_singlerange = np.nanstd(dh.data)/np.sqrt(neff_singlerange)
    stderr_doublerange = np.nanstd(dh.data)/np.sqrt(neff_doublerange)
    list_stderr_singlerange.append(stderr_singlerange)
    list_stderr_doublerange.append(stderr_doublerange)

# %%
# We add an empirical error based on intensive Monte-Carlo sampling ("patches" method) to validate our results
# (Dehecq et al. (2020), Hugonnet et al., in prep). This method is implemented in :func:`xdem.spatialstats.patches_method`.
# Here, we sample fewer areas to avoid for the patches method to run over long processing times, increasing from areas
# of 5 pixels to areas of 10000 pixels exponentially.
areas_emp = [10 * 400 * 2 ** i for i in range(10)]
for area_emp in areas_emp:

    #  First, sample intensively circular patches of a given area, and derive the mean elevation differences
    df_patches = xdem.spatialstats.patches_method(dh.data.data, gsd=dh.res[0], area=area_emp, nmax=200, random_state=42)
    # Second, estimate the dispersion of the means of each patch, i.e. the standard error of the mean
    stderr_empirical = np.nanstd(df_patches['nanmedian'].values)
    list_stderr_empirical.append(stderr_empirical)

fig, ax = plt.subplots()
plt.plot(np.asarray(areas)/1000000, list_stderr_singlerange, label='Single-range spherical model')
plt.plot(np.asarray(areas)/1000000, list_stderr_doublerange, label='Double-range spherical model')
plt.scatter(np.asarray(areas_emp)/1000000, list_stderr_empirical, label='Empirical estimate', color='black', marker='x')
plt.xlabel('Averaging area (km²)')
plt.ylabel('Uncertainty in the mean elevation difference (m)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# %%
# Using a single-range variogram highly underestimates the measurement error integrated over an area, by over a factor
# of ~100 for large surface areas. Using a double-range variogram brings us closer to the empirical error.
#
# **But, in this case, the error is still too small. Why?**
# The small size of the sampling area against the very large range of the noise implies that we might not verify the
# assumption of second-order stationarity (see :ref:`spatialstats`). Longer range correlations might be omitted by
# our analysis, due to the limits of the variogram sampling. In other words, a small part of the variance could be
# fully correlated over a large part of the grid: a vertical bias.
#
# As a first guess for this, let's examine the difference between mean and median to gain some insight on the central
# tendency of our sample:

diff_med_mean = np.nanmean(dh.data.data)-np.nanmedian(dh.data.data)
print('Difference mean/median: {:.3f} meters.'.format(diff_med_mean))

# %%
# If we now express it as a percentage of the dispersion:

print('{:.1f} % of STD.'.format(diff_med_mean/np.nanstd(dh.data.data)*100))

# %%
# There might be a significant bias of central tendency, i.e. almost fully correlated measurement error across the grid.
# If we assume that around 5% of the variance is fully correlated, and re-calculate our elevation measurement errors
# accordingly.

list_stderr_doublerange_plus_fullycorrelated = []
for area in areas:

    # For a double-range model
    neff_doublerange = xdem.spatialstats.neff_circ(area, [(params2[0], 'Sph', params2[1]),
                                                          (params2[2], 'Sph', params2[3])])

    # About 5% of the variance might be fully correlated, the other 95% has the random part that we quantified
    stderr_fullycorr = np.sqrt(0.05*np.nanvar(dh.data))
    stderr_doublerange = np.sqrt(0.95*np.nanvar(dh.data))/np.sqrt(neff_doublerange)
    list_stderr_doublerange_plus_fullycorrelated.append(stderr_fullycorr + stderr_doublerange)

fig, ax = plt.subplots()
plt.plot(np.asarray(areas)/1000000, list_stderr_singlerange, label='Single-range spherical model')
plt.plot(np.asarray(areas)/1000000, list_stderr_doublerange, label='Double-range spherical model')
plt.plot(np.asarray(areas)/1000000, list_stderr_doublerange_plus_fullycorrelated,
         label='5% fully correlated,\n 95% double-range spherical model')
plt.scatter(np.asarray(areas_emp)/1000000, list_stderr_empirical, label='Empirical estimate', color='black', marker='x')
plt.xlabel('Averaging area (km²)')
plt.ylabel('Uncertainty in the mean elevation difference (m)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# %%
# Our final estimation is now very close to the empirical error estimate.
#
# Some take-home points:
#   1. Long-range correlations are very important to reliably estimate measurement errors integrated in space, even if they have a small partial sill i.e. correlated variance,
#   2. Ideally, the grid must only contain correlation patterns significantly smaller than the grid size to verify second-order stationarity. Otherwise, be wary of small biases of central tendency, i.e. fully correlated measurement errors!
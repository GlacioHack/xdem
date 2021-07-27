"""
Spatial correlation of elevation measurement errors
===================================================

Digital elevation models have elevation measurement errors that can vary with terrain or instrument-related variables
(see :ref:`sphx_glr_auto_examples_plot_nonstationary_error.py`), but those measurement errors are also often
`spatially correlated <https://en.wikipedia.org/wiki/Spatial_analysis#Spatial_auto-correlation>`_.
While many DEM studies have been using short-range `variogram <https://en.wikipedia.org/wiki/Variogram>`_ models to
estimate the correlation of elevation measurement errors (e.g., `Howat et al. (2008) <https://doi.org/10.1029/2008GL034496>`_
, `Wang and Kääb (2015) <https://doi.org/10.3390/rs70810117>`_), recent studies show that variograms of multiple ranges
provide more realistic estimates of spatial correlation for many DEMs (e.g., `Dehecq et al. (2020) <https://doi.org/10.3389/feart.2020.566802>`_
, `Hugonnet et al. (2021) <https://doi.org/10.1038/s41586-021-03436-z>`_).

Quantifying the spatial correlation in elevation measurement errors is essential to integrate measurement errors over
an area of interest (e.g, to estimate the error of a mean or sum of samples). Once the spatial correlations are quantified,
several methods exist the approximate the measurement error in space (`Rolstad et al. (2009) <https://doi.org/10.3189/002214309789470950>`_
, Hugonnet et al. (in prep)). Further details are availale in :ref:`spatialstats`.

Here, we show an example in which we estimate spatially integrated elevation measurement errors for a DEM difference of
Longyearbyen glacier. We first quantify the spatial correlations using :func:`xdem.spatialstats.sample_multirange_empirical_variogram`
based on routines of `scikit-gstat <https://mmaelicke.github.io/scikit-gstat/index.html>`_. We then model the empirical variogram
using a sum of variogram models using :func:`xdem.spatialstats.fit_model_sum_vgm`.
Finally, we integrate the variogram models for varying surface areas to estimate the spatially integrated elevation
measurement errors for this DEM difference.

"""
# sphinx_gallery_thumbnail_number = 5
import matplotlib.pyplot as plt
import numpy as np
import xdem
import geoutils as gu

# %%
# We start by loading example files including a difference of DEMs at Longyearbyen glacier, the reference DEM later used to derive
# several terrain attributes, and the outlines to rasterize a glacier mask.
# Prior to differencing, the DEMs were aligned using :class:`xdem.coreg.NuthKaab` as shown in
# the :ref:`sphx_glr_auto_examples_plot_nuth_kaab.py` example. We later refer to those elevation differences as *dh*.

dh = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask_glacier = glacier_outlines.create_mask(dh)

# %%
# We remove values on glacier terrain, to use only stable terrain as a proxy for the elevation measurement errors
dh.data[mask_glacier] = np.nan

# %%
# We estimate the average per-pixel elevation measurement error on stable terrain, using both the standard deviation
# and normalized median absolute deviation
print('STD: {:.2f}'.format(np.nanstd(dh.data)))
print('NMAD: {:.2f}'.format(xdem.spatialstats.nmad(dh.data)))

# %%
# The two measures are quite similar which shows that, on average, there is a limited influence of outliers on the
# elevation differences. The precision per-pixel is, on average, :math:`\pm` 2.5 meters at the 1-sigma confidence level.
# Yet, the per-pixel precision is a limited metric to quantify the quality of the data to perform further spatial
# analysis.
# Let's plot the elevation differences to visually check the quality of the data.
plt.figure(figsize=(8, 5))
plt_extent = [
    dh.bounds.left,
    dh.bounds.right,
    dh.bounds.bottom,
    dh.bounds.top,
]
plt.imshow(dh.data.squeeze(), cmap="RdYlBu", vmin=-4, vmax=4, extent=plt_extent)
cbar = plt.colorbar()
cbar.set_label('Elevation differences (m)')
plt.show()


# %%
# We see that the residual elevation differences on stable terrain are clearly not random. The positive and negative
# differences (blue and red, respectively) seem correlated over large distances. **This needs to be quantified to
# estimate elevation measurement errors for a sum, or average of elevation difference a certain surface area**.
# Additionally, the elevation differences are still polluted by unrealistically large elevation differences near
# glaciers, probably because the glacier inventory is more recent than the data, and the outlines are too small.
# To remedy this, let's filter elevation difference outliers outside 4 NMAD.
dh.data[np.abs(dh.data) > 4 * xdem.spatialstats.nmad(dh.data)] = np.nan

# %%
# We plot the elevation differences after filtering.
plt.figure(figsize=(8, 5))
plt_extent = [
    dh.bounds.left,
    dh.bounds.right,
    dh.bounds.bottom,
    dh.bounds.top,
]
plt.imshow(dh.data.squeeze(), cmap="RdYlBu", vmin=-4, vmax=4, extent=plt_extent)
cbar = plt.colorbar()
cbar.set_label('Elevation differences (m)')
plt.show()

# %%
# To quantify the spatial correlation of the data, we sample an empirical variogram which calculates the covariance
# between the elevation differences of pairs of pixels depending on their distance. This distance between pairs of
# pixels if referred to as spatial lag.
# To perform this effectively, we use methods providing efficient pairwise sampling methods for large grid data,
# encapsulated by :func:`xdem.spatialstats.sample_multirange_variogram`:
df = xdem.spatialstats.sample_multirange_variogram(
    values=dh.data, gsd=dh.res[0], subsample=50, runs=30, nrun=10)

# %%
# We can now plot the empirical variogram:
xdem.spatialstats.plot_vgm(df)

# %%
# With this plot, it is hard to conclude anything.
# Properly visualizing the empirical variogram is one of the most important step. With grid data, we expect short-range
# correlations close to the resolution of the grid (~20-200 meters), but also possibly longer range correlation due to
# instrument noise or alignment issues (~1-50 km) (Hugonnet et al., in prep).

# To better visualize the variogram, we can either change the axis to log-scale, but this might make it more difficult
# to later compare to variogram models.
# Another solution is to split the variogram plot into subpanels, each with its own linear scale:
xdem.spatialstats.plot_vgm(df, xscale='log')
xdem.spatialstats.plot_vgm(df, xscale_range_split=[100, 1000, 10000])

# %%
# We identify a short-range (short spatial lag) correlation, likely due to effects of resolution, which has a large
# partial sill (correlated variance), meaning that the elevation measurement errors are strongly correlated until a
# range of ~200 m.
# We also identify a longer range correlation, with a smaller partial sill, meaning the part of the elevation
# measurement errors remain correlated over a longer distance.
# To show the difference between accounting only for the most noticeable, short-range correlation, and the long-range
# correlation, we fit those empirical variogram with two different models: a single spherical model (one range), and
# the sum of two spherical models (two ranges).
# For this, we use :func:`xdem.spatialstats.fit_model_sum_vgm`:
fun, params1 = xdem.spatialstats.fit_model_sum_vgm(['Sph'], emp_vgm_df=df)

fun2, params2 = xdem.spatialstats.fit_model_sum_vgm(['Sph', 'Sph'], emp_vgm_df=df)
xdem.spatialstats.plot_vgm(df,list_fit_fun=[fun, fun2],list_fit_fun_label=['Single-range model', 'Double-range model'],
                           xscale='log')
xdem.spatialstats.plot_vgm(df,list_fit_fun=[fun, fun2],list_fit_fun_label=['Single-range model', 'Double-range model'],
                           xscale_range_split=[100, 1000, 10000])


# %%
# **The sum of two spherical models seems to fit better, by modelling a small additional partial sill at longer ranges.
# This additional partial sill (correlated variance) is quite small, and one could thus wonder that the influence on
# the estimation of elevation measurement error will also be small.**
# However, even if the correlated variance if small, long-range correlated signals have a large effect on measurement
# errors.
# Let's show how this affect the precision of the DEM integrated over a certain surface area, from pixel size to grid
# size, by spatially integrating the variogram model using :func:`xdem.spatialstats.neff_circ`. # We validate that the
# double-range model provides more realistic estimationss of the error based on intensive Monte-Carlo sampling
# ("patches" method) over the data grid (Dehecq et al. (2020), Hugonnet et al., in prep), which
# # is integrated in :func:`xdem.spatialstats.patches_method`.

# We store the integrated elevation measurement error for each area
list_stderr_singlerange, list_stderr_doublerange, list_stderr_empirical = ([] for i in range(3))

# Numerical and exact integration of variogram run fast, so we derive errors for many surface areas from squared pixel
# size to squared grid size, with same unit as the variogram (meters)
areas = np.linspace(20**2, 10000**2, 1000)
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

# Sample only feasable areas for patches method to avoid long processing times: increasing exponentially from areas of
# 5 pixels to areas of 10000 pixels
areas_emp = [10 * 400 * 2 ** i for i in range(10)]
for area_emp in areas_emp:

    # Empirically estimate standard error:
    # 1/ Sample intensively circular patches of a given area, and derive the mean elevation differences
    df_patches = xdem.spatialstats.patches_method(dh.data.data, gsd=dh.res[0], area=area_emp, nmax=200, verbose=True)
    # 2/ Estimate the dispersion of the patches means, i.e. the standard error of the mean
    stderr_empirical = np.nanstd(df_patches['mean'].values)
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

# %%
# Using a single-range variogram can underestimates the integrated elevation measurement error by a factor of ~100 for
# large surface areas, be careful to multi-range variogram !

list_stderr_doublerange_plus_fullycorrelated = []
for area in areas:

    # For a double-range model
    neff_doublerange = xdem.spatialstats.neff_circ(area, [(params2[0], 'Sph', params2[1]),
                                                          (params2[2], 'Sph', params2[3])])

    # About 10% of the variance might be fully correlated, the other 90% has the random part that we quantified
    stderr_fullycorr = np.sqrt(0.1*np.nanvar(dh.data))
    stderr_doublerange = np.sqrt(0.9*np.nanvar(dh.data))/np.sqrt(neff_doublerange)
    list_stderr_doublerange_plus_fullycorrelated.append(stderr_fullycorr + stderr_doublerange)

fig, ax = plt.subplots()
plt.plot(np.asarray(areas)/1000000, list_stderr_singlerange, label='Single-range spherical model')
plt.plot(np.asarray(areas)/1000000, list_stderr_doublerange, label='Double-range spherical model')
plt.plot(np.asarray(areas)/1000000, list_stderr_doublerange_plus_fullycorrelated,
         label='10% fully correlated,\n 90% double-range spherical model')
plt.scatter(np.asarray(areas_emp)/1000000, list_stderr_empirical, label='Empirical estimate', color='black', marker='x')
plt.xlabel('Averaging area (km²)')
plt.ylabel('Uncertainty in the mean elevation difference (m)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

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
# sphinx_gallery_thumbnail_number = 8
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
# We remove values on glacier terrain
dh.data[mask_glacier] = np.nan

# %%
# Let's plot the elevation differences
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
# We can see that the elevation differences are still polluted by unmasked glaciers: let's filter outliers outside 4 NMAD
dh.data[np.abs(dh.data) > 4 * xdem.spatialstats.nmad(dh.data)] = np.nan

# %%
# Let's plot the elevation differences after filtering
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
# Sample empirical variogram
df = xdem.spatialstats.sample_multirange_variogram(
    values=dh.data, gsd=dh.res[0], subsample=50, runs=100, nrun=10)

# Plot empirical variogram
# fig, ax = plt.subplots()
xdem.spatialstats.plot_vgm(df)
# fun, _ = xdem.spatialstats.fit_model_sum_vgm(['Sph'], df)
# fun2, _ = xdem.spatialstats.fit_model_sum_vgm(['Sph', 'Sph', 'Sph'], emp_vgm_df=df)




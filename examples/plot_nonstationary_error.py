"""
Non-stationarity of elevation measurement errors
================================================

Digital elevation models have a precision that can vary with terrain and instrument-related variables. However, quantifying
this precision is complex and non-stationarities, i.e. variability of the measurement error, has rarely been
accounted for, with only some studies that used arbitrary filtering thresholds on the slope or other variables (see :ref:`intro`).

Quantifying the non-stationarities in elevation measurement errors is essential to use stable terrain as a proxy for
assessing the precision on other types of terrain (Hugonnet et al., in prep) and allows to standardize the measurement
errors to reach a stationary variance, an assumption necessary for spatial statistics (see :ref`spatialstats`).

Here, we show an example in which we identify terrain-related non-stationarities for a DEM difference of Longyearbyen glacier.
We quantify those non-stationarities by `binning <https://en.wikipedia.org/wiki/Data_binning>`_ robustly
in N-dimension using :func:`xdem.spstats.nd_binning` and applying a N-dimensional interpolation
:func:`xdem.spstats.interp_nd_binning` to estimate a numerical function of the measurement error and derive the spatial
distribution of elevation measurement errors of the difference of DEMs.

**Reference**: `Hugonnet et al. (2021) <https://doi.org/10.1038/s41586-021-03436-z>`_, applied to the terrain slope
and quality of stereo-correlation (Equation 1, Extended Data Fig. 3a).
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

# We load the data
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
ddem = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
# Rasterize a glacier mask
mask_glacier = glacier_outlines.create_mask(ddem)
# Remove values on unstable terrain
ddem.data[mask_glacier] = np.nan

# %%
# We use the reference DEM to derive terrain variables such as slope, aspect, curvature that we'll use to explore potential
# non-stationarities in elevation measurement error.

# We compute the slope, aspect, and both plan and profile curvatures
slope, aspect, planc, profc = \
    xdem.terrain.get_terrain_attribute(dem=ref_dem.data,
                                       attribute=['slope','aspect', 'planform_curvature', 'profile_curvature'],
                                       resolution=ref_dem.res)

# %%
# We use :func:`xdem.spstats.nd_binning` to perform N-dimensional binning on all those terrain variables, with uniform
# bin length divided by 30. We use the :ref:`spatial_stats_nmad` as a robust measure of `statistical dispersion <https://en.wikipedia.org/wiki/Statistical_dispersion>`_.

df = xdem.spstats.nd_binning(values=ddem.data,list_var=[slope, aspect, planc, profc],
                             list_var_names=['slope','aspect','planc','profc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=30)

# %%
# We obtain a dataframe with the 1D binning results for each variable, the 2D binning results for all combinations of
# variables and the N-D (here 4D) binning with all variables.
# We can now visualize the results of the 1D binning of the computed NMAD of elevation differences with each variable
# using :func:`xdem.spstats.plot_1d_binning`.
# We can start with the slope that has been long known to be related to the elevation measurement error (e.g.,
# `Toutin (2002) <https://doi.org/10.1109/TGRS.2002.802878>`_).
xdem.spstats.plot_1d_binning(df, 'slope', 'nmad', 'Slope (degrees)', 'NMAD of dh (m)')

# %%
# We identify a clear variability, with the dispersion estimated from the NMAD increasing from ~2 meters for nearly flat
# slopes to above 12 meters for slopes steeper than 50°.
# In statistical terms, such a variability of `variance <https://en.wikipedia.org/wiki/Variance>`_ is referred as
# `heteroscedasticity <https://en.wikipedia.org/wiki/Heteroscedasticity>`_. Here we observe heteroscedastic elevation
# differences due to a non-stationarity of variance with the terrain slope.
#
# What about the aspect?

xdem.spstats.plot_1d_binning(df, 'aspect', 'nmad', 'Aspect (degrees)', 'NMAD of dh (m)')

# %%
# There is no variability with the aspect which shows a dispersion averaging 2-3 meters, i.e. that of the complete sample.
#
# What about the plan curvature?

xdem.spstats.plot_1d_binning(df, 'planc', 'nmad', 'Planform curvature (100 m$^{-1}$)', 'NMAD of dh (m)')

# %%
# The relation with the plan curvature remains ambiguous.
# We should better define our bins to avoid sampling bins with too many or too few samples. For this, we can partition
# the data in quantiles in :func:`xdem.spstats.nd_binning`.
# Note: we need a higher number of bins to work with quantiles and still resolve the edges of the distribution. Thus, as
# with many dimensions the N dimensional bin size increases exponentially, we avoid binning all variables at the same
# time and instead bin one at a time.
# We define 1000 quantile bins of size 0.001 (equivalent to 0.1% percentile bins) for the profile curvature:

df = xdem.spstats.nd_binning(values=ddem.data,list_var=[profc], list_var_names=['profc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(profc,0.001*i) for i in range(1001)]])
xdem.spstats.plot_1d_binning(df, 'profc', 'nmad', 'Profile curvature (100 m$^{-1}$)', 'NMAD of dh (m)')

# %%
# We now clearly identify the variability with the profile curvature, from 2 meters for low curvatures to above 4 meters
# for higher positive or negative curvature.
# What about the role of the plan curvature?

df = xdem.spstats.nd_binning(values=ddem.data,list_var=[planc], list_var_names=['planc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(planc,0.001*i) for i in range(1001)]])
xdem.spstats.plot_1d_binning(df, 'planc', 'nmad', 'Planform curvature (100 m$^{-1}$)', 'NMAD of dh (m)')

# %%
# The plan curvature shows a similar relation. Those are symmetrical with 0, and almost equal for both types of curvature.
# To simplify the analysis, we here combine those curvatures into the maximum absolute curvature:

# Derive maximum absolute curvature
maxc = np.maximum(np.abs(planc),np.abs(profc))
df = xdem.spstats.nd_binning(values=ddem.data,list_var=[maxc], list_var_names=['maxc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(maxc,0.002*i) for i in range(501)]])
xdem.spstats.plot_1d_binning(df, 'maxc', 'nmad', 'Maximum absolute curvature (100 m$^{-1}$)', 'NMAD of dh (m)')

# %%
# Here's our simplified relation! We now have both slope and maximum absolute curvature with clear variability of
# the elevation measurement error.
#
# **But, one might wonder: high curvatures might occur more often around steep slopes than flat slope,
# so what if those two dependencies are actually one and the same?**
#
# We need to explore the variability with both slope and curvature at the same time:

df = xdem.spstats.nd_binning(values=ddem.data,list_var=[slope, maxc], list_var_names=['slope','maxc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=30)

xdem.spstats.plot_2d_binning(df, 'slope', 'maxc', 'nmad', 'Slope (degrees)', 'Maximum absolute curvature (100 m$^{-1}$)', 'NMAD of dh (m)')

# %%
# We can see that part of the variability seems to be independent, but with the uniform bins it is hard to tell much
# more.
#
# If we use custom quantiles for both binning variables, and adjust the plot scale:

custom_bin_slope = np.unique([np.quantile(slope,0.05*i) for i in range(20)]
                             + [np.quantile(slope,0.95+0.02*i) for i in range(2)]
                             + [np.quantile(slope,0.98 + 0.005*i) for i in range(3)]
                             + [np.quantile(slope,0.995 + 0.001*i) for i in range(6)])

custom_bin_curvature =  np.unique([np.quantile(maxc,0.05*i) for i in range(20)]
                             + [np.quantile(maxc,0.95+0.02*i) for i in range(2)]
                             + [np.quantile(maxc,0.98 + 0.005*i) for i in range(3)]
                             + [np.quantile(maxc,0.995 + 0.001*i) for i in range(6)])

df = xdem.spstats.nd_binning(values=ddem.data,list_var=[slope, maxc], list_var_names=['slope','maxc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[custom_bin_slope,custom_bin_curvature])
xdem.spstats.plot_2d_binning(df, 'slope', 'maxc', 'nmad', 'Slope (degrees)', 'Maximum absolute curvature (100 m$^{-1}$)', 'NMAD of dh (m)', scale_var_2='log', vmin=2, vmax=10)


# %%
# We identify clearly that the two variables have an independent effect on the precision, with
#
# - *high curvatures and flat slopes* that have larger errors than *low curvatures and flat slopes*
# - *steep slopes and low curvatures* that have larger errors than *low curvatures and flat slopes* as well
#
# We also identify that, steep slopes (> 40°) only correspond to high curvature, while the opposite is not true, hence
# the importance of mapping the variability in two dimensions.
#
# Now we need to account for the non-stationarities identified. For this, the simplest approach is a numerical
# approximation i.e. a piecewise linear interpolation/extrapolation based on the binning results.
# To ensure that only robust statistic values are used in the interpolation, we set a ``min_count`` value at 30 samples.

slope_curv_to_dh_err = xdem.spstats.interp_nd_binning(df,list_var_names=['slope','maxc'],statistic='nmad',min_count=30)

# %%
# The output is an interpolant function of slope and curvature that we can use to estimate the elevation measurement
# error at any point.
#
# For instance, estimate the elevation measurement error for different points:

for s, c in [(0.,0.1), (50.,0.1), (0.,20.), (50.,20.)]:
    print('Elevation measurement error for slope of {0:.0f} degrees, '
          'curvature of {1:.2f} m-1: {2:.1f}'.format(s, c/100, slope_curv_to_dh_err((s,c)))+ ' meters.')

# %%
# The same function can be used to estimate the spatial distribution of the elevation measurement error over the area:
dh_err = slope_curv_to_dh_err((slope, maxc))

plt.figure(figsize=(8, 5))
plt_extent = [
    ref_dem.bounds.left,
    ref_dem.bounds.right,
    ref_dem.bounds.bottom,
    ref_dem.bounds.top,
]
plt.imshow(dh_err.squeeze(), cmap="Reds", vmin=2, vmax=8, extent=plt_extent)
cbar = plt.colorbar()
cbar.set_label('Elevation measurement error (m)')
plt.show()






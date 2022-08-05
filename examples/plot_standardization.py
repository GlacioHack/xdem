"""
Standardization for stable terrain as proxy
===========================================

Digital elevation models have both a precision that can vary with terrain or instrument-related variables, and
a spatial correlation of measurement errors that can be due to effects of resolution, processing or instrument noise.
Accouting for non-stationarities in elevation measurement errors is essential to use stable terrain as a proxy to
infer the precision on other types of terrain (Hugonnet et al., in prep) and reliably use spatial statistics (see
:ref:`spatialstats`).

Here, we show an example to use standardization of the data based on the terrain-dependent nonstationarity in measurement
error (see :ref:`sphx_glr_auto_examples_plot_nonstationary_error.py`) and combine it with an analysis of spatial
correlation (see :ref:`sphx_glr_auto_examples_plot_vgm_error.py`) to derive spatially integrated errors for specific
spatial ensembles.

**Reference**: `Hugonnet et al. (2021) <https://doi.org/10.1038/s41586-021-03436-z>`_, applied to the terrain slope
and quality of stereo-correlation (Equation 1, Extended Data Fig. 3a).
"""
# sphinx_gallery_thumbnail_number = 4
import matplotlib.pyplot as plt
import numpy as np
import xdem
import geoutils as gu
from xdem.spatialstats import nmad

# %%
# We start by estimating the non-stationarities and deriving a terrain-dependent measurement error as a function of both
# slope and maximum curvature, as shown in the  :ref:`sphx_glr_auto_examples_plot_nonstationary_error.py` example.

# Load the data
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dh = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask_glacier = glacier_outlines.create_mask(dh)

# Compute the slope and maximum curvature
slope, planc, profc = \
    xdem.terrain.get_terrain_attribute(dem=ref_dem.data,
                                       attribute=['slope', 'planform_curvature', 'profile_curvature'],
                                       resolution=ref_dem.res)

# Remove values on unstable terrain
dh_arr = dh.data[~mask_glacier]
slope_arr = slope[~mask_glacier]
planc_arr = planc[~mask_glacier]
profc_arr = profc[~mask_glacier]
maxc_arr = np.maximum(np.abs(planc_arr),np.abs(profc_arr))

# Remove large outliers
dh_arr[np.abs(dh_arr) > 4 * xdem.spatialstats.nmad(dh_arr)] = np.nan

# Define bins for 2D binning
custom_bin_slope = np.unique(np.concatenate([np.nanquantile(slope_arr,np.linspace(0,0.95,20)),
                                             np.nanquantile(slope_arr,np.linspace(0.96,0.99,5)),
                                             np.nanquantile(slope_arr,np.linspace(0.991,1,10))]))

custom_bin_curvature = np.unique(np.concatenate([np.nanquantile(maxc_arr,np.linspace(0,0.95,20)),
                                             np.nanquantile(maxc_arr,np.linspace(0.96,0.99,5)),
                                             np.nanquantile(maxc_arr,np.linspace(0.991,1,10))]))

# Perform 2D binning to estimate the measurement error with slope and maximum curvature
df = xdem.spatialstats.nd_binning(values=dh_arr, list_var=[slope_arr, maxc_arr], list_var_names=['slope', 'maxc'],
                                  statistics=['count', np.nanmedian, nmad],
                                  list_var_bins=[custom_bin_slope,custom_bin_curvature])

# Estimate an interpolant of the measurement error with slope and maximum curvature
slope_curv_to_dh_err = xdem.spatialstats.interp_nd_binning(df, list_var_names=['slope', 'maxc'], statistic='nmad', min_count=30)
maxc = np.maximum(np.abs(profc), np.abs(planc))

# Estimate a measurement error per pixel
dh_err = slope_curv_to_dh_err((slope, maxc))

# %%
# Using the measurement error estimated for each pixel, we standardize the elevation differences by applying
# a simple division:

z_dh = dh.data/dh_err

# %%
# We remove values on glacierized terrain and large outliers.
z_dh.data[mask_glacier] = np.nan
z_dh.data[np.abs(z_dh.data)>4] = np.nan

# %%
# We perform a scale-correction for the standardization, to ensure that the standard deviation of the data is exactly 1.
print('Standard deviation before scale-correction: {:.1f}'.format(nmad(z_dh.data)))
scale_fac_std = nmad(z_dh.data)
z_dh = z_dh/scale_fac_std
print('Standard deviation after scale-correction: {:.1f}'.format(nmad(z_dh.data)))

plt.figure(figsize=(8, 5))
plt_extent = [
    ref_dem.bounds.left,
    ref_dem.bounds.right,
    ref_dem.bounds.bottom,
    ref_dem.bounds.top,
]
ax = plt.gca()
glacier_outlines.ds.plot(ax=ax, fc='none', ec='tab:gray')
ax.plot([], [], color='tab:gray', label='Glacier 1990 outlines')
plt.imshow(z_dh.squeeze(), cmap="RdYlBu", vmin=-3, vmax=3, extent=plt_extent)
cbar = plt.colorbar()
cbar.set_label('Standardized elevation differences (m)')
plt.legend(loc='lower right')
plt.show()

# %%
# Now, we can perform an analysis of spatial correlation as shown in the :ref:`sphx_glr_auto_examples_plot_vgm_error.py`
# example, by estimating a variogram and fitting a sum of two models.
df_vgm = xdem.spatialstats.sample_empirical_variogram(values=z_dh.data.squeeze(), gsd=dh.res[0], subsample=1000,
                                                      n_variograms=10, random_state=42)

func_sum_vgm, params_vgm = xdem.spatialstats.fit_sum_model_variogram(['Gaussian', 'Spherical'], empirical_variogram=df_vgm)
xdem.spatialstats.plot_vgm(df_vgm, xscale_range_split=[100, 1000, 10000], list_fit_fun=[func_sum_vgm],
                           list_fit_fun_label=['Standardized double-range variogram'])

# %%
# With standardized input, the variogram should converge towards one. With the input data close to a stationary
# variance, the variogram will be more robust as it won't be affected by changes in variance due to terrain- or
# instrument-dependent variability of measurement error. The variogram should only capture changes in variance due to
# spatial correlation.

# %%
# **How to use this standardized spatial analysis to compute final uncertainties?**
#
# Let's take the example of two glaciers of similar size: Svendsenbreen and Medalsbreen, which are respectively
# north and south-facing. The south-facing Medalsbreen glacier is subject to more sun exposure, and thus should be
# located in higher slopes, with possibly higher curvatures.

svendsen_shp = gu.Vector(glacier_outlines.ds[glacier_outlines.ds['NAME'] == 'Svendsenbreen'])
svendsen_mask = svendsen_shp.create_mask(dh)

medals_shp = gu.Vector(glacier_outlines.ds[glacier_outlines.ds['NAME'] == 'Medalsbreen'])
medals_mask = medals_shp.create_mask(dh)

plt.figure(figsize=(8, 5))
ax = plt.gca()
plt_extent = [
    ref_dem.bounds.left,
    ref_dem.bounds.right,
    ref_dem.bounds.bottom,
    ref_dem.bounds.top,
]
plt.imshow(slope.squeeze(), cmap="Blues", vmin=0, vmax=40, extent=plt_extent)
cbar = plt.colorbar(ax=ax)
cbar.set_label('Slope (degrees)')
svendsen_shp.ds.plot(ax=ax, fc='none', ec='tab:olive', lw=2)
medals_shp.ds.plot(ax=ax, fc='none', ec='tab:gray', lw=2)
plt.plot([],[], color='tab:olive', label='Medalsbreen')
plt.plot([], [], color='tab:gray', label='Svendsenbreen')
plt.legend(loc='lower left')
plt.show()

print('Average slope of Svendsenbreen glacier: {:.1f}'.format(np.nanmean(slope[svendsen_mask])))
print('Average maximum curvature of Svendsenbreen glacier: {:.3f}'.format(np.nanmean(maxc[svendsen_mask])))

print('Average slope of Medalsbreen glacier: {:.1f}'.format(np.nanmean(slope[medals_mask])))
print('Average maximum curvature of Medalsbreen glacier : {:.1f}'.format(np.nanmean(maxc[medals_mask])))

# %%
# We calculate the number of effective samples for each glacier based on the variogram
svendsen_neff = xdem.spatialstats.neff_circular_approx_numerical(area=svendsen_shp.ds.area.values[0], params_vgm=params_vgm)

medals_neff = xdem.spatialstats.neff_circular_approx_numerical(area=medals_shp.ds.area.values[0], params_vgm=params_vgm)

print('Number of effective samples of Svendsenbreen glacier: {:.1f}'.format(svendsen_neff))
print('Number of effective samples of Medalsbreen glacier: {:.1f}'.format(medals_neff))

# %%
# Due to the long-range spatial correlations affecting the elevation differences, both glacier have a similar, low
# number of effective samples. This transcribes into a large standardized integrated error.

svendsen_z_err = 1/np.sqrt(svendsen_neff)
medals_z_err = 1/np.sqrt(medals_neff)

print('Standardized integrated error of Svendsenbreen glacier: {:.1f}'.format(svendsen_z_err))
print('Standardized integrated error of Medalsbreen glacier: {:.1f}'.format(medals_z_err))

# %%
# Finally, we destandardize the spatially integrated errors based on the measurement error dependent on slope and
# maximum curvature. This yields the uncertainty into the mean elevation change for each glacier.

# Destandardize the uncertainty
fac_svendsen_dh_err = scale_fac_std * np.nanmean(dh_err[svendsen_mask])
fac_medals_dh_err = scale_fac_std * np.nanmean(dh_err[medals_mask])
svendsen_dh_err = fac_svendsen_dh_err * svendsen_z_err
medals_dh_err = fac_medals_dh_err * medals_z_err

# Derive mean elevation change
svendsen_dh = np.nanmean(dh.data[svendsen_mask])
medals_dh = np.nanmean(dh.data[medals_mask])

# Plot the result
plt.figure(figsize=(8, 5))
ax = plt.gca()
plt.imshow(dh.data.squeeze(), cmap="RdYlBu", vmin=-50, vmax=50, extent=plt_extent)
cbar = plt.colorbar(ax=ax)
cbar.set_label('Elevation differences (m)')
svendsen_shp.ds.plot(ax=ax, fc='none', ec='tab:olive', lw=2)
medals_shp.ds.plot(ax=ax, fc='none', ec='tab:gray', lw=2)
plt.plot([],[], color='tab:olive', label='Svendsenbreen glacier')
plt.plot([],[], color='tab:gray', label='Medalsbreen glacier')
ax.text(svendsen_shp.ds.centroid.x.values[0], svendsen_shp.ds.centroid.y.values[0]-1500,
        '{:.2f} \n$\\pm$ {:.2f}'.format(svendsen_dh, svendsen_dh_err), color='tab:olive', fontweight='bold',
        va='top', ha='center', fontsize=12)
ax.text(medals_shp.ds.centroid.x.values[0], medals_shp.ds.centroid.y.values[0]+2000,
        '{:.2f} \n$\\pm$ {:.2f}'.format(medals_dh, medals_dh_err), color='tab:gray', fontweight='bold',
        va='bottom', ha='center', fontsize=12)
plt.legend(loc='lower left')
plt.show()

# %%
# Because of slightly higher slopes and curvatures, the final uncertainty for Medalsbreen is larger by about 10%.
# The differences between the mean terrain slope and curvatures of stable terrain and those of glaciers is quite limited
# on Svalbard. In high moutain terrain, such as the Alps or Himalayas, the difference between stable terrain and glaciers,
# and among glaciers, would be much larger.
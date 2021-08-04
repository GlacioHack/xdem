"""
Non-stationarity of elevation measurement errors
================================================

Digital elevation models have a precision that can vary with terrain and instrument-related variables. However, quantifying
this precision is complex and non-stationarities, i.e. variability of the measurement error, has rarely been
accounted for, with only some studies that used arbitrary filtering thresholds on the slope or other variables (see :ref:`intro`).

Quantifying the non-stationarities in elevation measurement errors is essential to use stable terrain as a proxy for
assessing the precision on other types of terrain (Hugonnet et al., in prep) and allows to standardize the measurement
errors to reach a stationary variance, an assumption necessary for spatial statistics (see :ref:`spatialstats`).

Here, we show an example in which we identify terrain-related non-stationarities for a DEM difference of Longyearbyen glacier.
We quantify those non-stationarities by `binning <https://en.wikipedia.org/wiki/Data_binning>`_ robustly
in N-dimension using :func:`xdem.spatialstats.nd_binning` and applying a N-dimensional interpolation
:func:`xdem.spatialstats.interp_nd_binning` to estimate a numerical function of the measurement error and derive the spatial
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
# We start by estimating the non-stationarities and deriving a terrain-dependent measurement error as shown in
# the :ref:`sphx_glr_auto_examples_plot_nonstationarity_error.py` example.

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

# We remove values on unstable terrain
dh_arr = dh.data[~mask_glacier]
slope_arr = slope[~mask_glacier]
planc_arr = planc[~mask_glacier]
profc_arr = profc[~mask_glacier]
maxc_arr = np.maximum(np.abs(planc_arr),np.abs(profc_arr))

dh_arr[np.abs(dh_arr) > 4 * xdem.spatialstats.nmad(dh_arr)] = np.nan


custom_bin_slope = np.unique(np.concatenate([np.quantile(slope_arr,np.linspace(0,0.95,20)),
                                             np.quantile(slope_arr,np.linspace(0.96,0.99,5)),
                                             np.quantile(slope_arr,np.linspace(0.991,1,10))]))

custom_bin_curvature = np.unique(np.concatenate([np.quantile(maxc_arr,np.linspace(0,0.95,20)),
                                             np.quantile(maxc_arr,np.linspace(0.96,0.99,5)),
                                             np.quantile(maxc_arr,np.linspace(0.991,1,10))]))

df = xdem.spatialstats.nd_binning(values=dh_arr, list_var=[slope_arr, maxc_arr], list_var_names=['slope', 'maxc'],
                                  statistics=['count', np.nanmedian, np.nanstd],
                                  list_var_bins=[custom_bin_slope,custom_bin_curvature])

# Estimate an interpolant of the measurement error with slope and maximum curvature
slope_curv_to_dh_err = xdem.spatialstats.interp_nd_binning(df, list_var_names=['slope', 'maxc'], statistic='nanstd', min_count=30)
maxc = np.maximum(np.abs(profc), np.abs(planc))

# Estimated measurement error per pixel
dh_err = slope_curv_to_dh_err((slope, maxc))

# %%
# Standardization of the elevation differences
z_dh = dh.data/dh_err
z_dh[mask_glacier] = np.nan
fac_std = np.nanstd(z_dh)
z_dh = z_dh/fac_std

df_vgm = xdem.spatialstats.sample_empirical_variogram(
    values=z_dh.squeeze(), gsd=dh.res[0], subsample=50, runs=30, n_variograms=10, estimator='cressie', random_state=42)

fun, params = xdem.spatialstats.fit_sum_model_variogram(['Sph', 'Sph'], empirical_variogram=df_vgm)
xdem.spatialstats.plot_vgm(df_vgm, xscale_range_split=[100, 1000, 10000], list_fit_fun=[fun], list_fit_fun_label=['Standardized double-range variogram'])

# %%
# Let's compute the uncertainty for two glaciers
plog_shp = gu.Vector(glacier_outlines.ds[glacier_outlines.ds['IDENT'] == 13622.100000000000364])
plog_mask = plog_shp.create_mask(dh)

southfacing_shp = gu.Vector(glacier_outlines.ds[glacier_outlines.ds['IDENT'] == 13623])
southfacing_mask = southfacing_shp.create_mask(dh)

print('Average slope of Plogbreen: {:.1f}'.format(np.nanmean(slope[plog_mask])))
print('Average maximum curvature of Plogbreen: {:.3f}'.format(np.nanmean(maxc[plog_mask])))

print('Average slope of unnamed south-facing glacier: {:.1f}'.format(np.nanmean(slope[southfacing_mask])))
print('Average maximum curvature of unnamed south-facing glacier : {:.1f}'.format(np.nanmean(maxc[southfacing_mask])))

# %%
plog_neff = xdem.spatialstats.neff_circ(plog_shp.ds['Shape_Area'].values[0],  [(params[0], 'Sph', params[1]),
                                                   (params[2], 'Sph', params[3])])

southfacing_neff = xdem.spatialstats.neff_circ(southfacing_shp.ds['Shape_Area'].values[0],  [(params[0], 'Sph', params[1]),
                                                   (params[2], 'Sph', params[3])])

plog_z_err = 1/np.sqrt(plog_neff)
southfacing_z_err = 1/np.sqrt(southfacing_neff)

# %%
fac_plog_dh_err = fac_std * np.nanmean(dh_err[plog_mask])
fac_southfacing_dh_err = fac_std * np.nanmean(dh_err[southfacing_mask])

# %%
plog_dh_err = fac_plog_dh_err * plog_z_err
southfacing_dh_err = fac_southfacing_dh_err * southfacing_z_err

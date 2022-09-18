"""Code example for spatial statistics"""
import geoutils as gu
import numpy as np

import xdem

# Load data
dh = gu.georaster.Raster(xdem.examples.get_path("longyearbyen_ddem"))
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
glacier_mask = gu.geovector.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(dh)

slope = xdem.terrain.get_terrain_attribute(ref_dem, attribute=['slope'])

# Keep only stable terrain data
dh.set_mask(mask)
dh_arr = gu.spatial_tools.get_array_and_mask(dh)[0]
slope_arr = gu.spatial_tools.get_array_and_mask(slope)[0]

# Subsample to run the snipped code faster
indices = gu.spatial_tools.subsample_raster(dh_arr, subsample=10000, return_indices=True,
                                            random_state=42)
dh_arr = dh_arr[indices]
slope_arr = slope_arr[indices]

# Estimate the measurement error by bin of slope, using the NMAD as robust estimator
df_ns = xdem.spatialstats.nd_binning(dh_arr, list_var=[slope_arr], list_var_names=['slope'], statistics=['count', xdem.spatialstats.nmad])

# Derive a numerical function of the measurement error
err_dh = xdem.spatialstats.interp_nd_binning(df_ns, list_var_names=['slope'])

# Standardize the data
z_dh = dh_arr / err_dh(slope_arr)

# Sample empirical variogram
df_vgm = xdem.spatialstats.sample_empirical_variogram(values=dh, subsample=10, random_state=42)

# Fit sum of double-range spherical model
func_sum_vgm, params_variogram_model = xdem.spatialstats.fit_sum_model_variogram(list_models = ['Gaussian', 'Spherical'],
                                                                                 empirical_variogram=df_vgm)

# Calculate the area-averaged uncertainty with these models
neff = xdem.spatialstats.number_effective_samples(area = 1000, params_variogram_model=params_variogram_model)

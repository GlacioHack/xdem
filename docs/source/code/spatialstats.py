"""Code example for spatial statistics"""
import xdem
import geoutils as gu
import numpy as np

# Load data
ddem = gu.georaster.Raster(xdem.examples.get_path("longyearbyen_ddem"))
glacier_mask = gu.geovector.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(ddem)

# Get slope for non-stationarity
slope = xdem.coreg.calculate_slope_and_aspect(ddem.data)[0]

# Keep only stable terrain data
ddem.data[mask] = np.nan

# Get non-stationarities by bins
df_ns = xdem.spatialstats.nd_binning(ddem.data.ravel(), list_var=[slope.ravel()], list_var_names=['slope'])

# Sample empirical variogram
df_vgm = xdem.spatialstats.sample_multirange_empirical_variogram(dh=ddem.data, nsamp=1000, nrun=20, nproc=10, maxlag=10000)

# Fit single-range spherical model
fun, coefs = xdem.spatialstats.fit_model_sum_vgm(['Sph'], emp_vgm_df=df_vgm)

# Fit sum of triple-range spherical model
fun2, coefs2 = xdem.spatialstats.fit_model_sum_vgm(['Sph', 'Sph', 'Sph'], emp_vgm_df=df_vgm)

# Calculate the area-averaged uncertainty with these models
list_vgm = [(coefs[2*i],'Sph',coefs[2*i+1]) for i in range(int(len(coefs)/2))]
neff = xdem.spatialstats.neff_circ(1, list_vgm)




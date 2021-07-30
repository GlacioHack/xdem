"""Code example for spatial statistics"""
import xdem
import geoutils as gu
import numpy as np

# Load data
dh = gu.georaster.Raster(xdem.examples.get_path("longyearbyen_ddem"))
glacier_mask = gu.geovector.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(dh)

# Get slope for non-stationarity
slope = xdem.terrain.get_terrain_attribute(dh.data, resolution=dh.res[0], attribute=['slope'])

# Keep only stable terrain data
dh.data[mask] = np.nan

# Estimate the measurement error by bin of slope, using the NMAD as robust estimator
df_ns = xdem.spatialstats.nd_binning(dh.data.ravel(), list_var=[slope.ravel()], list_var_names=['slope'],
                                     statistics=['count', xdem.spatialstats.nmad])

# Derive a numerical function of the measurement error
err_dh = xdem.spatialstats.interp_nd_binning(df_ns, list_var_names=['slope'])

# Sample empirical variogram
df_vgm = xdem.spatialstats.sample_empirical_variogram(values=dh.data, gsd=dh.res[0], subsample=50,
                                                      random_state=42, runs=10)
# Fit sum of double-range spherical model
fun, coefs = xdem.spatialstats.fit_sum_model_variogram(list_model=['Sph', 'Sph'], empirical_variogram=df_vgm)

# Calculate the area-averaged uncertainty with these models
list_vgm = []
for i in range(2):
    list_vgm.append((coefs[2 * i], "Sph", coefs[2 * i + 1]))
area = 1000
neff = xdem.spatialstats.neff_circ(area, list_vgm)




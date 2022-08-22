"""Code example for spatial statistics"""
import xdem
import geoutils as gu
import numpy as np

# Load data
dh = gu.georaster.Raster(xdem.examples.get_path("longyearbyen_ddem"))
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
glacier_mask = gu.geovector.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(dh)

# Get slope for non-stationarity
slope = xdem.terrain.get_terrain_attribute(dem=ref_dem.data, resolution=dh.res, attribute=['slope'])

# Keep only stable terrain data
dh.data[mask] = np.nan

# Estimate the measurement error by bin of slope, using the NMAD as robust estimator
df_ns = xdem.spatialstats.nd_binning(dh.data.ravel(), list_var=[slope.ravel()], list_var_names=['slope'],
                                     statistics=['count', xdem.spatialstats.nmad], list_var_bins=30)

xdem.spatialstats.plot_1d_binning(df_ns, 'slope', 'nmad', 'Slope (degrees)', 'Elevation measurement error (m)')
"""Code example for spatial statistics"""

import geoutils as gu

import xdem

# Load data
dh = gu.Raster(xdem.examples.get_path("longyearbyen_ddem"))
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
glacier_mask = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(dh)

# Get slope for non-stationarity
slope = xdem.terrain.get_terrain_attribute(dem=ref_dem, attribute=["slope"])

# Keep only stable terrain data
dh.load()
dh.set_mask(mask)

# Estimate the measurement error by bin of slope, using the NMAD as robust estimator
df_ns = xdem.spatialstats.nd_binning(
    dh.data.ravel(),
    list_var=[slope.data.ravel()],
    list_var_names=["slope"],
    statistics=["count", gu.stats.nmad],
    list_var_bins=30,
)

xdem.spatialstats.plot_1d_binning(df_ns, "slope", "nmad", "Slope (degrees)", "Random elevation error\n($1\\sigma$, m)")

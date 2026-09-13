"""Code example for spatial statistics"""

import geoutils as gu

import xdem

# Load data
dh = gu.Raster(xdem.examples.get_path("longyearbyen_ddem"))
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
glacier_mask = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(dh)

# Get slope for non-stationarity
slope = xdem.terrain.get_terrain_attribute(dem=ref_dem, attribute="slope")

# Keep only stable terrain data
dh.load()
dh.set_mask(mask)

# Estimate the measurement error by bin of slope, using the NMAD as robust estimator
df_ns = dh.grouped_stats({"slope": slope}, bins={"slope": 30}, statistics=[gu.stats.nmad], observed=False)
axes = gu.stats.plot_grouped_stats(df_ns, statistic="nmad")
axes["statistic"].set(xlabel="Slope (degrees)", ylabel="Random elevation error (1 sigma, m)")

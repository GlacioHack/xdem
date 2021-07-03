"""
Non-stationarities in measurement errors
========================================
Test
"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import pandas as pd
import xdem
import geoutils as gu

# %%
# **Example files**
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
ddem = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))

glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask_glacier = glacier_outlines.create_mask(ddem)

# Mask out unstable terrain
ddem.data[mask_glacier] = np.nan

# Get terrain variables
slope, aspect, planc, profc = \
    xdem.terrain.get_terrain_attribute(dem=ref_dem.data,
                                       attribute=['slope','aspect', 'planform_curvature', 'profile_curvature'],
                                       resolution=ref_dem.res)

# Look at possible non-stationarities with terrain variables
df = xdem.spstats.nd_binning(values=ddem.data,list_var=[slope, aspect, planc, profc],
                             list_var_names=['slope','aspect','planc','profc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=10)

# Let's look at each variable
xdem.spstats.plot_1d_binning(df, 'slope', 'nmad', 'Slope (degrees)', 'NMAD of elevation differences (m)')
xdem.spstats.plot_1d_binning(df, 'aspect', 'nmad', 'Aspect (degrees)', 'NMAD of elevation differences (m)')
xdem.spstats.plot_1d_binning(df, 'planc', 'nmad', 'Planform curvature (100 m$^{-1}$)', 'NMAD of elevation differences (m)')
xdem.spstats.plot_1d_binning(df, 'profc', 'nmad', 'Profile curvature (100 m$^{-1}$)', 'NMAD of elevation differences (m)')

# There is a clear dependency to slope, none clear with aspect, but it is ambiguous with the curvature
# We should better define our bins to avoid sampling bins with too many or too few samples
# For this, we can partition the data in quantile when calling nd_binning.
# Note: we need a higher number of bins to work with quantiles and estimate the edges of the distribution
# As with N dimensions the N dimensional bin size increases exponentially, we avoid binning all variables at the same
# and bin one by one
df = xdem.spstats.nd_binning(values=ddem.data,list_var=[slope], list_var_names=['slope'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(slope,0.002*i) for i in range(501)]])
xdem.spstats.plot_1d_binning(df, 'slope', 'nmad', 'Slope (degrees)', 'NMAD of elevation differences (m)')
df = xdem.spstats.nd_binning(values=ddem.data,list_var=[profc], list_var_names=['profc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(profc,0.005*i) for i in range(201)]])
xdem.spstats.plot_1d_binning(df, 'profc', 'nmad', 'Profile curvature (100 m$^{-1}$)', 'NMAD of elevation differences (m)')

df = xdem.spstats.nd_binning(values=ddem.data,list_var=[planc], list_var_names=['planc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(planc,0.005*i) for i in range(201)]])
xdem.spstats.plot_1d_binning(df, 'planc', 'nmad', 'Planform curvature (100 m$^{-1}$)', 'NMAD of elevation differences (m)')

# We see there is a clear relation with plan and profile curvatures, that is symmetrical and similar for both types of curvature.
# Thus, we can here use the maximum absolute curvature to simplify our analysis

# Derive maximum absolute curvature
maxc = np.maximum(np.abs(planc),np.abs(profc))
df = xdem.spstats.nd_binning(values=ddem.data,list_var=[maxc], list_var_names=['maxc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(maxc,0.002*i) for i in range(501)]])
xdem.spstats.plot_1d_binning(df, 'maxc', 'nmad', 'Maximum absolute curvature (100 m$^{-1}$)', 'NMAD of elevation differences (m)')

# There is indeed a clear relation with curvature as well !
# But, high curvatures might occur more often around steep slopes, so what if those dependencies are one and the same?
# We should explore the inter-dependency of slope and curvature

df = xdem.spstats.nd_binning(values=ddem.data,list_var=[slope, maxc], list_var_names=['slope','maxc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=10)

xdem.spstats.plot_2d_binning(df, 'slope', 'maxc', 'nmad', 'Slope (degrees)', 'Maximum absolute curvature (100 m$^{-1}$)', 'NMAD of dh (m)')

# We can't see much with uniform bins again, let's try to use quantiles for both binning variables, and *
# adjust the plot scale to display those quantiles properly.
df = xdem.spstats.nd_binning(values=ddem.data,list_var=[slope, maxc], list_var_names=['slope','maxc'],
                             statistics=['count',np.nanmedian,xdem.spstats.nmad],
                             list_var_bins=[[np.nanquantile(slope,0.05*i) for i in range(21)],[np.nanquantile(maxc,0.025*i) for i in range(41)]])
xdem.spstats.plot_2d_binning(df, 'slope', 'maxc', 'nmad', 'Slope (degrees)', 'Maximum absolute curvature (100 m$^{-1}$)', 'NMAD of dh (m)', scale_var_2='log', vmin=2, vmax=5)

# We can see that both variable have an effect on the precision:
# - high-curvature at low slopes have larger errors
# - high-slopes at low curvature have larger errors as well
# We should thus try to account for both of those dependencies.





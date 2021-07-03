"""
Non-stationarities in measurement errors
"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt

import numpy as np
import xdem
import geoutils as gu

# %%
# **Example files**
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
ddem = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))

glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask_glacier = glacier_outlines.create_mask(ddem)

# Plot
def plot_1d_binning(df, l)
ax, fig = plt.subplots()
plt.plot(df.)

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
                             statistics=['count',np.nanmedian,xdem.spstats.nmad])



"""
Nuth and Kääb coregistration
============================

Nuth and Kääb (`2011 <https:https://doi.org/10.5194/tc-5-271-2011>`_) coregistration allows for horizontal and vertical shifts to be estimated and corrected for.
In ``xdem``, this approach is implemented through the :class:`xdem.coreg.NuthKaab` class.

For more information about the approach, see :ref:`coregistration_nuthkaab`.
"""
import geoutils as gu
import numpy as np
import xdem

# %%
# **Example files**
reference_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_to_be_aligned = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# Create a stable ground mask (not glacierized) to mark "inlier data"
inlier_mask = ~glacier_outlines.create_mask(reference_dem)


# %%
# The DEM to be aligned (a 1990 photogrammetry-derived DEM) has some vertical and horizontal biases that we want to avoid.
# These can be visualized by plotting a change map:

diff_before = reference_dem - dem_to_be_aligned
diff_before.show(cmap="coolwarm_r", vmin=-10, vmax=10, cb_title="Elevation change (m)")


# %%
# Horizontal and vertical shifts can be estimated using :class:`xdem.coreg.NuthKaab`.
# First, the shifts are estimated, and then they can be applied to the data:

nuth_kaab = xdem.coreg.NuthKaab()

nuth_kaab.fit(reference_dem, dem_to_be_aligned, inlier_mask)

aligned_dem = nuth_kaab.apply(dem_to_be_aligned)

# %%
# Then, the new difference can be plotted to validate that it improved.

diff_after = reference_dem - aligned_dem
diff_after.show(cmap="coolwarm_r", vmin=-10, vmax=10, cb_title="Elevation change (m)")


# %%
# We compare the median and NMAD to validate numerically that there was an improvement (see :ref:`robuststats_meanstd`):
inliers_before = diff_before.data[inlier_mask].compressed()
med_before, nmad_before = np.median(inliers_before), xdem.spatialstats.nmad(inliers_before)

inliers_after = diff_after.data[inlier_mask].compressed()
med_after, nmad_after = np.median(inliers_after), xdem.spatialstats.nmad(inliers_after)

print(f"Error before: median = {med_before:.2f} - NMAD = {nmad_before:.2f} m")
print(f"Error after: median = {med_after:.2f} - NMAD = {nmad_after:.2f} m")

# %%
# In the plot above, one may notice a positive (blue) tendency toward the east.
# The 1990 DEM is a mosaic, and likely has a "seam" near there.
# :ref:`sphx_glr_advanced_examples_plot_blockwise_coreg.py` tackles this issue, using a nonlinear coregistration approach.

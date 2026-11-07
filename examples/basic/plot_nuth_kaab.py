"""
Nuth and Kääb coregistration
============================

The Nuth and Kääb coregistration corrects horizontal and vertical shifts, and is especially performant for precise
sub-pixel alignment in areas with varying slope.
In xDEM, this approach is implemented through the :class:`xdem.coreg.NuthKaab` class.

See also the :ref:`nuthkaab` section in feature pages.

**Reference:** `Nuth and Kääb (2011) <https:https://doi.org/10.5194/tc-5-271-2011>`_.
"""

import geoutils as gu
import numpy as np

import xdem

# %%
# We open example files.
reference_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_to_be_aligned = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# We create a stable ground mask (not glacierized) to mark "inlier data".
inlier_mask = ~glacier_outlines.create_mask(reference_dem)

# %%
# The DEM to be aligned (a 1990 photogrammetry-derived DEM) has some vertical and horizontal biases that we want to reduce.
# These can be visualized by plotting a change map:

diff_before = reference_dem - dem_to_be_aligned
diff_before.plot(cmap="RdYlBu", vmin=-10, vmax=10, cbar_title="Elevation change (m)")

# %%
# Horizontal and vertical shifts can be estimated using :class:`~xdem.coreg.NuthKaab`.
# The shifts are estimated then applied to the to-be-aligned elevation data:

nuth_kaab = xdem.coreg.NuthKaab()
aligned_dem = nuth_kaab.fit_and_apply(reference_dem, dem_to_be_aligned, inlier_mask)

# %%
# The shifts are stored in the affine metadata output

print([nuth_kaab.meta["outputs"]["affine"][s] for s in ["shift_x", "shift_y", "shift_z"]])

# %%
# Then, the new difference can be plotted to validate that it improved.

diff_after = reference_dem - aligned_dem
diff_after.plot(cmap="RdYlBu", vmin=-10, vmax=10, cbar_title="Elevation change (m)")

# %%
# We compare the median and NMAD to validate numerically that there was an improvement (see :ref:`robuststats-meanstd`):
inliers_before = diff_before[inlier_mask]
med_before, nmad_before = np.ma.median(inliers_before), gu.stats.nmad(inliers_before)

inliers_after = diff_after[inlier_mask]
med_after, nmad_after = np.ma.median(inliers_after), gu.stats.nmad(inliers_after)

print(f"Error before: median = {med_before:.2f} - NMAD = {nmad_before:.2f} m")
print(f"Error after: median = {med_after:.2f} - NMAD = {nmad_after:.2f} m")

# %%
# In the plot above, one may notice a positive (blue) tendency toward the east.
# The 1990 DEM is a mosaic, and likely has a "seam" near there.
# :ref:`sphx_glr_advanced_examples_plot_blockwise_coreg.py` tackles this issue, using a nonlinear coregistration approach.

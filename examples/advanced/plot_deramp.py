"""
Bias-correction with deramping
==============================

Deramping can help correct rotational or doming errors in elevation data.
In xDEM, this approach is implemented through the :class:`xdem.coreg.Deramp` class.

See also the :ref:`deramp` section in feature pages.
"""

import geoutils as gu
import numpy as np

import xdem

# %%
# We open example files.
reference_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_to_be_aligned = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# Create a stable ground mask (not glacierized) to mark "inlier data"
inlier_mask = ~glacier_outlines.create_mask(reference_dem)

# %%
# We visualize the patterns of error from the elevation differences.

diff_before = reference_dem - dem_to_be_aligned
diff_before.plot(cmap="RdYlBu", vmin=-10, vmax=10, cbar_title="Elevation differences (m)")


# %%
# A 2-D 3rd order polynomial is estimated, and applied to the data:

deramp = xdem.coreg.Deramp(poly_order=2)

corrected_dem = deramp.fit_and_apply(reference_dem, dem_to_be_aligned, inlier_mask=inlier_mask)

# %%
# Then, the new difference can be plotted.

diff_after = reference_dem - corrected_dem
diff_after.plot(cmap="RdYlBu", vmin=-10, vmax=10, cbar_title="Elevation differences (m)")


# %%
# We compare the median and NMAD to validate numerically that there was an improvement (see :ref:`robuststats-meanstd`):
inliers_before = diff_before[inlier_mask]
med_before, nmad_before = np.ma.median(inliers_before), gu.stats.nmad(inliers_before)

inliers_after = diff_after[inlier_mask]
med_after, nmad_after = np.ma.median(inliers_after), gu.stats.nmad(inliers_after)

print(f"Error before: median = {med_before:.2f} - NMAD = {nmad_before:.2f} m")
print(f"Error after: median = {med_after:.2f} - NMAD = {nmad_after:.2f} m")

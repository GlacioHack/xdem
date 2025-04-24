"""
Elevation error map
===================

Digital elevation models have a precision that can vary with terrain and instrument-related variables. Here, we
rely on a non-stationary spatial statistics framework to estimate and model this variability in elevation error,
using terrain slope and maximum curvature as explanatory variables, with stable terrain as an error proxy for moving
terrain.

**Reference:** `Hugonnet et al. (2022) <https://doi.org/10.1109/jstars.2022.3188922>`_.
"""

import geoutils as gu

# sphinx_gallery_thumbnail_number = 1
import xdem

# %%
# We load a difference of DEMs at Longyearbyen, already coregistered using :ref:`nuthkaab` as shown in
# the :ref:`sphx_glr_basic_examples_plot_nuth_kaab.py` example. We also load the reference DEM to derive terrain
# attributes and the glacier outlines here corresponding to moving terrain.
dh = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# %%
# We derive the terrain slope and maximum curvature from the reference DEM.
slope, maximum_curvature = xdem.terrain.get_terrain_attribute(ref_dem, attribute=["slope", "maximum_curvature"])

# %%
# Then, we run the pipeline for inference of elevation heteroscedasticity from stable terrain:
errors, df_binning, error_function = xdem.spatialstats.infer_heteroscedasticity_from_stable(
    dvalues=dh, list_var=[slope, maximum_curvature], list_var_names=["slope", "maxc"], unstable_mask=glacier_outlines
)

# %%
# The first output corresponds to the error map for the DEM (:math:`\pm` 1\ :math:`\sigma` level):
errors.plot(vmin=2, vmax=7, cmap="Reds", cbar_title=r"Elevation error (1$\sigma$, m)")

# %%
# The second output is the dataframe of 2D binning with slope and maximum curvature:
df_binning

# %%
# The third output is the 2D binning interpolant, i.e. an error function with the slope and maximum curvature
# (*Note: below we divide the maximum curvature by 100 to convert it in* m\ :sup:`-1` ):
for slope, maxc in [(0, 0), (40, 0), (0, 5), (40, 5)]:
    print(
        "Error for a slope of {:.0f} degrees and"
        " {:.2f} m-1 max. curvature: {:.1f} m".format(slope, maxc / 100, error_function((slope, maxc)))
    )

# %%
# This pipeline will not always work optimally with default parameters: spread estimates can be affected by skewed
# distributions, the binning by extreme range of values, some DEMs do not have any error variability with terrain (e.g.,
# terrestrial photogrammetry). **To learn how to tune more parameters and use the subfunctions, see the gallery example:**
# :ref:`sphx_glr_advanced_examples_plot_heterosc_estimation_modelling.py`!

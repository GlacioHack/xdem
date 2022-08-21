"""
Elevation error map
===================

Digital elevation models have a precision that can vary with terrain and instrument-related variables. Here, we apply
the framework of `Hugonnet et al. (2022) <https://doi.org/10.1109/jstars.2022.3188922>`_ to estimate and model this
variability in elevation error, using terrain slope and maximum curvature as explanatory variables and stable terrain
as an error proxy for moving terrain.

**References**: `Hugonnet et al. (2022) <https://doi.org/10.1109/jstars.2022.3188922>`_. See in particular Figure 4.

Errors in elevation difference can be converted in elevation errors following Equation 7 (equal if other source of much
higher precision) or Equation 8 (divided by sqrt(2) if the two sources are of same precision).
"""
# sphinx_gallery_thumbnail_number = 1
import xdem
import geoutils as gu

# %%
# We load a difference of DEMs at Longyearbyen, already coregistered using :ref:`coregistration_nuthkaab` as shown in
# the :ref:`sphx_glr_auto_examples_plot_nuth_kaab.py` example. We also load the reference DEM to derive terrain
# attributes and the glacier outlines here corresponding to moving terrain.
dh = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# %%
# We derive the terrain slope and maximum curvature from the reference DEM.
slope, maximum_curvature = xdem.terrain.get_terrain_attribute(ref_dem, attribute=['slope', 'maximum_curvature'])

# %%
# Then, we run the pipeline for inference of elevation heteroscedasticity from stable terrain:
errors, df_binning, error_function = \
    xdem.spatialstats.infer_heteroscedasticy_from_stable(dvalues=dh, list_var=[slope, maximum_curvature],
                                                         list_var_names=['slope', 'maxc'],
                                                         unstable_mask=glacier_outlines)

# %%
# The first output corresponds to the error map for the DEM (1-sigma):
errors.show(vmin=2, vmax=7, cmap='Reds', cb_title='Elevation error (1$\sigma$)')

# %%
# The second output is the dataframe of 2D binning with slope and maximum curvature:
df_binning

# %%
# The third output is the 2D binning interpolant, i.e. an error function with the slope and maximum curvature
# (*Note: below we multiply the maximum curvature by 100 to convert it in m-1*):
print('Error for a slope of {:.0f} degrees and {:.0f} m-1 max. curvature: {:.1f} m'.format(0, 0, error_function((0, 0))))
print('Error for a slope of {:.0f} degrees and {:.0f} m-1 max. curvature: {:.1f} m'.format(40, 0, error_function((40, 0))))
print('Error for a slope of {:.0f} degrees and {:.0f} m-1 max. curvature: {:.1f} m'.format(0, 100*5, error_function((0, 5))))

# %%
# This pipeline will not always work optimally with default parameters: spread estimates can be affected by skewed
# distributions, the binning by extreme range of values, some DEMs do not have any error variability with terrain (e.g.,
# terrestrial photogrammetry). **To learn how to tune more parameters and use the subfunctions, see the gallery example:**
# :ref:`sphx_glr_auto_examples_heterosc_estimation_modelling.py`!
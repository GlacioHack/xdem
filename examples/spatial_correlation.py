"""
Spatial correlation of errors
=============================

Digital elevation models have errors that are spatially correlated due to instrument or processing effects. Here, we apply
the framework of `Hugonnet et al. (2022) <https://doi.org/10.1109/jstars.2022.3188922>`_ to estimate and model this
spatial correlation in elevation error, using a sum of variogram forms to model this correlation, and stable terrain
as an error proxy for moving terrain.

**References**: `Hugonnet et al. (2022) <https://doi.org/10.1109/jstars.2022.3188922>`_. See in particular Figure 5.
"""
# sphinx_gallery_thumbnail_number = 1
import xdem
import geoutils as gu

# %%
# We load a difference of DEMs at Longyearbyen, already coregistered using :ref:`coregistration_nuthkaab` as shown in
# the :ref:`sphx_glr_auto_examples_plot_nuth_kaab.py` example. We also load the glacier outlines here corresponding to
# moving terrain.
dh = xdem.DEM(xdem.examples.get_path("longyearbyen_ddem"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# %%
# Then, we run the pipeline for inference of elevation heteroscedasticity from stable terrain (*Note: we pass a
# random_state argument to ensure a fixed random subsampling in this example, useful for result reproducibility*):
df_empirical_variogram, df_model_params, spatial_corr_function = \
    xdem.spatialstats.infer_spatial_correlation_from_stable(dvalues=dh, list_models=['Gaussian', 'Spherical'],
                                                            unstable_mask=glacier_outlines, random_state=42)

# %%
# The first output corresponds to the dataframe of the empirical variogram, by default estimated using Dowd's estimator
# and the circular sampling scheme of `skgstat.RasterEquidistantMetricSpace` (Fig. S13 of Hugonnet et al. (2022)):
df_empirical_variogram

# %%
# The second output is the dataframe of optimized model parameters for a sum of gaussian and spherical models:
df_model_params

# %%
# The third output is the spatial correlation function with spatial lags, derived from the variogram:
print('Errors are correlated at {:.1f}% for a {:,.0f} m spatial lag'.format(spatial_corr_function(0)*100, 0))
print('Errors are correlated at {:.1f}% for a {:,.0f} m spatial lag'.format(spatial_corr_function(100)*100, 100))
print('Errors are correlated at {:.1f}% for a {:,.0f} m spatial lag'.format(spatial_corr_function(1000)*100, 1000))
print('Errors are correlated at {:.1f}% for a {:,.0f} m spatial lag'.format(spatial_corr_function(10000)*100, 10000))
print('Errors are correlated at {:.1f}% for a {:,.0f} m spatial lag'.format(spatial_corr_function(30000)*100, 30000))

# %%
# We can plot the empirical variogram and its model on a non-linear X-axis to identify the multi-scale correlations.
xdem.spatialstats.plot_variogram(df=df_empirical_variogram,
                                 list_fit_fun=[xdem.spatialstats.get_variogram_model_func(df_model_params)],
                                 xlabel='Spatial lag (m)',
                                 ylabel='Variance of\nelevation differences (m)',
                                 xscale_range_split=[100, 1000])

# %%
# This pipeline will not always work optimally with default parameters: variogram sampling is more robust with a lot of
# samples but takes long computing times, and the fitting might require multiple tries for forms and possibly bounds
# and first guesses to help the least-squares optimization. **To learn how to tune more parameters and use the
# subfunctions, see the gallery example:** :ref:`sphx_glr_auto_examples_variogram_estimation_modelling.py`!
"""Plot example for variogram"""
import matplotlib.pyplot as plt

import geoutils as gu
import xdem
import numpy as np

# load data
ddem = gu.georaster.Raster(xdem.examples.get_path("longyearbyen_ddem"))
glacier_mask = gu.geovector.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask = glacier_mask.create_mask(ddem)

# remove glacier data
ddem.data[mask] = np.nan

# ensure the figures are reproducible
np.random.seed(42)

# sample empirical variogram
df = xdem.spatialstats.sample_multirange_empirical_variogram(dh=ddem.data, gsd=ddem.res[0], nsamp=1000, nrun=20, maxlag=4000)

fun, _ = xdem.spatialstats.fit_model_sum_vgm(['Sph'], df)
fun2, _ = xdem.spatialstats.fit_model_sum_vgm(['Sph', 'Sph', 'Sph'], emp_vgm_df=df)
xdem.spatialstats.plot_vgm(df, list_fit_fun=[fun, fun2], list_fit_fun_label=['Spherical model', 'Sum of three spherical models'])

plt.show()


import xdem
import geoutils as gu
import numpy as np

# load diff and mask
xdem.examples.download_longyearbyen_examples(overwrite=False)

reference_raster = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
to_be_aligned_raster = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
glacier_mask = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
inlier_mask = ~glacier_mask.create_mask(reference_raster)

nuth_kaab = xdem.coreg.NuthKaab()
nuth_kaab.fit(reference_raster.data, to_be_aligned_raster.data,
              inlier_mask=inlier_mask, transform=reference_raster.transform)
aligned_raster = nuth_kaab.apply(to_be_aligned_raster.data, transform=reference_raster.transform)

ddem = gu.Raster.from_array((reference_raster.data - aligned_raster),
                            transform=reference_raster.transform, crs=reference_raster.crs)
mask = glacier_mask.create_mask(ddem)

# ddem is a difference of DEMs
x, y = ddem.coords(offset='center')
coords = np.dstack((x.flatten(), y.flatten())).squeeze()

# Sample empirical variogram
df = xdem.spstats.sample_multirange_empirical_variogram(dh=ddem.data, nsamp=1000, nrun=20, nproc=10, maxlag=10000)

# Fit single-range spherical model
fun, coefs = xdem.spstats.fit_model_sum_vgm(['Sph'], df)

# Fit sum of triple-range spherical model
fun2, coefs2 = xdem.spstats.fit_model_sum_vgm(['Sph', 'Sph', 'Sph'], emp_vgm_df=df)

# Calculate the area-averaged uncertainty with these models
list_vgm = [(coefs[2*i],'Sph',coefs[2*i+1]) for i in range(int(len(coefs)/2))]
neff = xdem.spstats.neff_circ(1,list_vgm)




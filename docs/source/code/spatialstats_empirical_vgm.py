import matplotlib.pyplot as plt

import geoutils as gu
import xdem
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

# extract coordinates
x, y = ddem.coords(offset='center')
coords = np.dstack((x.flatten(), y.flatten())).squeeze()

# ensure the figures are reproducible
np.random.seed(42)

# sample empirical variogram
df = xdem.spstats.sample_multirange_empirical_variogram(dh=ddem.data, nsamp=1000, nrun=20, maxlag=10000)

# plot empirical variogram
xdem.spstats.plot_vgm(df)
plt.show()



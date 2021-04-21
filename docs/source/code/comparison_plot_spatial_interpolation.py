"""Plot an example of spatial interpolation of randomly generated errors."""
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np

import xdem

xdem.examples.download_longyearbyen_examples(overwrite=False)

dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
outlines_1990 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])

ddem = xdem.dDEM(
    xdem.spatial_tools.subtract_rasters(dem_2009, dem_1990, resampling_method="nearest"),
    start_time=np.datetime64("1990-08-01"),
    end_time=np.datetime64("2009-08-01")
)
# The example DEMs are void-free, so let's make some random voids.
ddem.data.mask = np.zeros_like(ddem.data, dtype=bool)  # Reset the mask
# Introduce 50000 nans randomly throughout the dDEM.
ddem.data.mask.ravel()[np.random.choice(ddem.data.size, 50000, replace=False)] = True

ddem.interpolate(method="linear")

ylim = (300, 100)
xlim = (800, 1050)

plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.imshow(ddem.data.squeeze(), cmap="coolwarm_r", vmin=-50, vmax=50)
plt.ylim(ylim)
plt.xlim(xlim)
plt.axis("off")
plt.title("dDEM with random voids")
plt.subplot(122)
plt.imshow(ddem.filled_data.squeeze(), cmap="coolwarm_r", vmin=-50, vmax=50)
plt.ylim(ylim)
plt.xlim(xlim)
plt.axis("off")
plt.title("Linearly interpolated dDEM")


plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import xdem

xdem.examples.download_longyearbyen_examples()

dem = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])

slope = xdem.terrain.slope(dem.data, resolution=dem.res)

plt.imshow(slope.squeeze(), cmap="Reds")
cbar = plt.colorbar()
cbar.set_label("Slope (°)")

plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.show()

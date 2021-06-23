import matplotlib.pyplot as plt
import xdem

xdem.examples.download_longyearbyen_examples()

dem = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])

aspect = xdem.terrain.aspect(dem.data)

plt.imshow(aspect.squeeze(), cmap="jet")
cbar = plt.colorbar()
cbar.set_label("Aspect (°)")

plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt
import xdem

dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))

slope = xdem.terrain.slope(dem.data, resolution=dem.res)

plt.figure(figsize=(8, 5))
plt.imshow(slope.squeeze(), cmap="Reds")
cbar = plt.colorbar()
cbar.set_label("Slope (°)")

plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.show()

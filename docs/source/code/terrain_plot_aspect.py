import matplotlib.pyplot as plt
import xdem

dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))

aspect = xdem.terrain.aspect(dem.data)

plt.figure(figsize=(8, 5))
plt.imshow(aspect.squeeze(), cmap="twilight")
cbar = plt.colorbar()
cbar.set_label("Aspect (°)")

plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.show()

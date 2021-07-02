import matplotlib.pyplot as plt
import xdem

dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))

curvature = xdem.terrain.profile_curvature(dem.data, resolution=dem.res)

plt.figure(figsize=(8, 5))
plt.imshow(curvature.squeeze(), cmap="RdGy_r", vmin=-1, vmax=1, interpolation="bilinear")
cbar = plt.colorbar()
cbar.set_label("Profile curvature (100 / m)")

plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.show()

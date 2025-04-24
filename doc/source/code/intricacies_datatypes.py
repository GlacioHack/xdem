"""Plot example of elevation data types for guide page."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import xdem

# Open reference DEM and crop to small area
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
ref_dem = ref_dem.crop(
    (ref_dem.bounds.left, ref_dem.bounds.bottom, ref_dem.bounds.left + 1000, ref_dem.bounds.bottom + 1000)
)

# Get point cloud with 100 points
ref_epc = ref_dem.to_pointcloud(subsample=100, random_state=42)

f, ax = plt.subplots(2, 2, squeeze=False, sharex=True, sharey=True)
# Plot 1: DEM
ax[0, 0].set_title("DEM")
ref_dem.plot(cmap="terrain", ax=ax[0, 0], vmin=280, vmax=420, cbar_title="Elevation (m)")
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.gca().set_aspect("equal")

# Plot 2: EPC
ax[0, 1].set_title("Elevation\npoint cloud")
point = ref_epc.plot(column="b1", cmap="terrain", ax=ax[0, 1], vmin=280, vmax=420, cbar_title="Elevation (m)")
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.gca().set_aspect("equal")

# Plot 3: TIN
ax[1, 1].set_title("Elevation TIN")
triang = matplotlib.tri.Triangulation(ref_epc.geometry.x.values, ref_epc.geometry.y.values)
ax[1, 1].triplot(triang, color="gray", marker=".")
scat = ax[1, 1].scatter(
    ref_epc.geometry.x.values, ref_epc.geometry.y.values, c=ref_epc["b1"].values, cmap="terrain", vmin=280, vmax=420
)
plt.colorbar(mappable=scat, ax=ax[1, 1], label="Elevation (m)", pad=0.02)
ax[1, 1].set_xticklabels([])
ax[1, 1].set_yticklabels([])
ax[1, 1].set_aspect("equal")

# Plot 4: Contour
ax[1, 0].set_title("Elevation contour")
coords = ref_dem.coords(grid=False)
cont = ax[1, 0].contour(
    np.flip(coords[0]), coords[1], np.flip(ref_dem.get_nanarray()), levels=15, cmap="terrain", vmin=280, vmax=420
)
plt.colorbar(mappable=cont, ax=ax[1, 0], label="Elevation (m)", pad=0.02)
ax[1, 0].set_xticklabels([])
ax[1, 0].set_yticklabels([])
ax[1, 0].set_aspect("equal")

plt.suptitle("Types of elevation data")

plt.tight_layout()
plt.show()

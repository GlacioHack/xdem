import matplotlib.pyplot as plt
import xdem

dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))

hillshade = xdem.terrain.hillshade(dem.data, resolution=dem.res, azimuth=315., altitude=45.)

plt.imshow(hillshade.squeeze(), cmap="Greys_r", vmin=0, vmax=255)
plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.show()

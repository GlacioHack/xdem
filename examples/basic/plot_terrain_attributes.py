"""
Terrain attributes
==================

Terrain attributes generated from a DEM have a multitude of uses for analytic and visual purposes.
Here is an example of how to generate these products.

For more information, see the :ref:`terrain-attributes` feature page.

**References:** `Horn (1981) <https://ieeexplore.ieee.org/document/1456186>`_ (slope, aspect, hillshade),
`Zevenbergen and Thorne (1987) <http://dx.doi.org/10.1002/esp.3290120107>`_ (curvature),
`Riley et al. (1999) <http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf>`_ (terrain
ruggedness index), `Jenness (2004) <https://doi.org/10.2193/0091-7648(2004)032%5B0829:CLSAFD%5D2.0.CO;2>`_ (rugosity).
"""

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt

import xdem

# %%
# We load the example data.

dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))

# %%
# We generate multiple terrain attribute at once (more efficient computationally as some depend on each other).

attributes = xdem.terrain.get_terrain_attribute(
    dem.data,
    resolution=dem.res,
    attribute=["hillshade", "slope", "aspect", "curvature", "terrain_ruggedness_index", "rugosity"],
)

plt.figure(figsize=(8, 6.5))

plt_extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]

cmaps = ["Greys_r", "Reds", "twilight", "RdGy_r", "Purples", "YlOrRd"]
labels = ["Hillshade", "Slope (°)", "Aspect (°)", "Curvature (100 / m)", "Terrain Ruggedness Index", "Rugosity"]
vlims = [(None, None) for i in range(6)]
vlims[3] = [-2, 2]

for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.imshow(attributes[i].squeeze(), cmap=cmaps[i], extent=plt_extent, vmin=vlims[i][0], vmax=vlims[i][1])
    cbar = plt.colorbar()
    cbar.set_label(labels[i])
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

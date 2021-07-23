"""
Terrain attributes
==================

Terrain attributes generated from a DEM have a multitude of uses for analytic and visual purposes.
Here is an example of how to generate these products.

For more information, see the :ref:`terrain_attributes` chapter.
"""

import matplotlib.pyplot as plt

import xdem

# %%
# **Example data**

dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))


def plot_attribute(attribute, cmap, label=None, vlim=None):
    plt.figure(figsize=(8, 5))

    vlims = {"vmin": -vlim, "vmax": vlim} if vlim is not None else {}
    plt.imshow(
        attribute.squeeze(),
        cmap=cmap,
        extent=[dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top],
        **vlims
    )
    if label is not None:
        cbar = plt.colorbar()
        cbar.set_label(label)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.show()


# %%
# Slope
# -----

slope = xdem.terrain.slope(dem.data, resolution=dem.res)

plot_attribute(slope, "Reds", "Slope (°)")

# %%
# Aspect
# ------

aspect = xdem.terrain.aspect(dem.data)

plot_attribute(aspect, "twilight", "Aspect (°)")

# %%
# Hillshade
# ---------

hillshade = xdem.terrain.hillshade(dem.data, resolution=dem.res, azimuth=315.0, altitude=45.0)

plot_attribute(hillshade, "Greys_r")

# %%
# Curvature
# ---------

curvature = xdem.terrain.curvature(dem.data, resolution=dem.res)

plot_attribute(curvature, "RdGy_r", "Curvature (100 / m)", vlim=1)

# %%
# Planform curvature
# ------------------

planform_curvature = xdem.terrain.planform_curvature(dem.data, resolution=dem.res)

plot_attribute(planform_curvature, "RdGy_r", "Planform curvature (100 / m)", vlim=1)

# %%
# Profile curvature
# -----------------
profile_curvature = xdem.terrain.profile_curvature(dem.data, resolution=dem.res)

plot_attribute(profile_curvature, "RdGy_r", "Profile curvature (100 / m)", vlim=1)

# %%
# Generating multiple attributes at once
# --------------------------------------

slope, aspect, hillshade = xdem.terrain.get_terrain_attribute(
    dem.data, attribute=["slope", "aspect", "hillshade"], resolution=dem.res
)

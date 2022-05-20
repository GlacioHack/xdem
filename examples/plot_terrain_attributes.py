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

    add_cb = True if label is not None else False

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    vlims = {"vmin": -vlim, "vmax": vlim} if vlim is not None else {}
    attribute.show(
        ax=ax,
        cmap=cmap,
        add_cb=add_cb,
        cb_title=label,
        **vlims
    )

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.show()


# %%
# Slope
# -----

slope = xdem.terrain.slope(dem)

plot_attribute(slope, "Reds", "Slope (°)")

# %%
# Note that all functions also work with numpy array as inputs, if resolution is specified

slope = xdem.terrain.slope(dem.data, resolution=dem.res)

# %%
# Aspect
# ------

aspect = xdem.terrain.aspect(dem)

plot_attribute(aspect, "twilight", "Aspect (°)")

# %%
# Hillshade
# ---------

hillshade = xdem.terrain.hillshade(dem, azimuth=315.0, altitude=45.0)

plot_attribute(hillshade, "Greys_r")

# %%
# Curvature
# ---------

curvature = xdem.terrain.curvature(dem)

plot_attribute(curvature, "RdGy_r", "Curvature (100 / m)", vlim=1)

# %%
# Planform curvature
# ------------------

planform_curvature = xdem.terrain.planform_curvature(dem)

plot_attribute(planform_curvature, "RdGy_r", "Planform curvature (100 / m)", vlim=1)

# %%
# Profile curvature
# -----------------
profile_curvature = xdem.terrain.profile_curvature(dem)

plot_attribute(profile_curvature, "RdGy_r", "Profile curvature (100 / m)", vlim=1)

# %%
# Generating multiple attributes at once
# --------------------------------------

slope, aspect, hillshade = xdem.terrain.get_terrain_attribute(
    dem, attribute=["slope", "aspect", "hillshade"]
)

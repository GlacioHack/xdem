"""
DEM subtraction
===============

Subtracting one DEM with another should be easy!
This is why ``xdem`` (with functionality from `geoutils <https://geoutils.readthedocs.io>`_) allows directly using the ``-`` or ``+`` operators on :class:`xdem.DEM` objects.

Before DEMs can be compared, they need to be reprojected/resampled/cropped to fit the same grid.
The :func:`xdem.DEM.reproject` method takes care of this.

"""
import xdem


# %%

dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))

# %%
# We can print the information about the DEMs for a "sanity check"

print(dem_2009)
print(dem_1990)

# %%
# In this particular case, the two DEMs are already on the same grid (they have the same bounds, resolution and coordinate system).
# If they don't, we need to reproject one DEM to fit the other.
# :func:`xdem.DEM.reproject` is a multi-purpose method that ensures a fit each time:

_ = dem_1990.reproject(dem_2009)

# %%
# Oops!
# ``xdem`` just warned us that ``dem_1990`` did not need reprojection, but we asked it to anyway.
# To hide this prompt, add ``.reproject(..., silent=True)``.
# By default, :func:`xdem.DEM.reproject` uses "bilinear" resampling (assuming resampling is needed).
# Other options are "nearest" (fast but inaccurate), "cubic_spline", "lanczos" and others.
# See `geoutils.Raster.reproject() <https://geoutils.readthedocs.io/en/latest/api.html#geoutils.georaster.Raster.reproject>`_ and `rasterio.enums.Resampling <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for more information about reprojection.
#
# Now, we are ready to generate the dDEM:

ddem = dem_2009 - dem_1990

print(ddem)

# %%
# It is a new :class:`xdem.DEM` instance, loaded in memory.
# Let's visualize it:

ddem.show(cmap="coolwarm_r", vmin=-20, vmax=20, cb_title="Elevation change (m)")

# %%
# For missing values, ``xdem`` provides a number of interpolation methods which are shown in the other examples.

# %%
# Saving the output to a file is also very simple

ddem.save("temp.tif")

# %%
# ... and that's it!

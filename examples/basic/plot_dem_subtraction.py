"""
DEM differencing
================

Subtracting a DEM with another one should be easy.

xDEM allows to use any operator on :class:`xdem.DEM` objects, such as :func:`+<operator.add>` or :func:`-<operator.sub>` as well as most NumPy functions
while respecting nodata values and checking that georeferencing is consistent. This functionality is inherited from `GeoUtils' Raster class <https://geoutils.readthedocs.io>`_.

Before DEMs can be compared, they need to be reprojected to the same grid and have the same 3D CRSs. The :func:`~xdem.DEM.reproject` and :func:`~xdem.DEM.to_vcrs` methods are used for this.

"""

import geoutils as gu

import xdem

# %%
# We load two DEMs near Longyearbyen, Svalbard.

dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))

# %%
# We can print the information about the DEMs for a "sanity check".

dem_2009.info()
dem_1990.info()

# %%
# In this particular case, the two DEMs are already on the same grid (they have the same bounds, resolution and coordinate system).
# If they don't, we need to reproject one DEM to fit the other using :func:`xdem.DEM.reproject`:

dem_1990 = dem_1990.reproject(dem_2009)

# %%
# Oops!
# GeoUtils just warned us that ``dem_1990`` did not need reprojection. We can hide this output with ``silent``.
# By default, :func:`~xdem.DEM.reproject` uses "bilinear" resampling (assuming resampling is needed).
# Other options are detailed at `geoutils.Raster.reproject() <https://geoutils.readthedocs.io/en/latest/api.html#geoutils.raster.Raster.reproject>`_ and `rasterio.enums.Resampling <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_.
#
# We now compute the difference by simply substracting, passing ``stats=True`` to :func:`xdem.DEM.info` to print statistics.

ddem = dem_2009 - dem_1990

ddem.info(stats=True)

# %%
# It is a new :class:`~xdem.DEM` instance, loaded in memory.
# Let's visualize it, with some glacier outlines.

# Load the outlines
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
glacier_outlines = glacier_outlines.crop(ddem, clip=True)
ddem.plot(cmap="RdYlBu", vmin=-20, vmax=20, cbar_title="Elevation differences (m)")
glacier_outlines.plot(ref_crs=ddem, fc="none", ec="k")

# %%
# And we save the output to file.

ddem.save("temp.tif")

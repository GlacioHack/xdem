"""Example code for the DEM comparison chapter (it's made like this to test the syntax)."""
#######################
# SECTION: Example data
#######################
from datetime import datetime

import geoutils as gu
import numpy as np

import xdem


# Load a reference DEM from 2009
dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"), datetime=datetime(2009, 8, 1))
# Load a DEM from 1990
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"), datetime=datetime(1990, 8, 1))
# Load glacier outlines from 1990.
glaciers_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
glaciers_2010 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines_2010"))

# Make a dictionary of glacier outlines where the key represents the associated date.
outlines = {
    datetime(1990, 8, 1): glaciers_1990,
    datetime(2009, 8, 1): glaciers_2010,
}

# Fake a future DEM to have a time-series of three DEMs
dem_2060 = dem_2009.copy()
# Assume that all glacier values will be 30 m lower than in 2009
dem_2060.data[glaciers_2010.create_mask(dem_2060)] -= 30
dem_2060.datetime = datetime(2060, 8, 1)

##############################
# SECTION: Subtracting rasters
##############################

dem1, dem2 = dem_2009, dem_1990

ddem_data = dem1.data - dem2.data
# If we want to inherit the georeferencing information:
ddem_raster = xdem.DEM.from_array(ddem_data, dem1.transform, dem2.crs)

# TEXT HERE

ddem_raster = xdem.spatial_tools.subtract_rasters(dem1, dem2)

#############################
# SECTION: dDEM interpolation
#############################

ddem = xdem.dDEM(
    raster=xdem.spatial_tools.subtract_rasters(dem_2009, dem_1990),
    start_time=dem_1990.datetime,
    end_time=dem_2009.datetime
)

# The example DEMs are void-free, so let's make some random voids.
ddem.data.mask = np.zeros_like(ddem.data, dtype=bool)  # Reset the mask
# Introduce 50000 nans randomly throughout the dDEM.
ddem.data.mask.ravel()[np.random.choice(ddem.data.size, 50000, replace=False)] = True

# SUBSECTION: Linear spatial interpolation

ddem.interpolate(method="linear")

# SUBSECTION: Local hypsometric interpolation

ddem.interpolate(method="local_hypsometric", reference_elevation=dem_2009, mask=glaciers_1990)

# SUBSECTION: Regional hypsometric interpolation

ddem.interpolate(method="regional_hypsometric", reference_elevation=dem_2009, mask=glaciers_1990)

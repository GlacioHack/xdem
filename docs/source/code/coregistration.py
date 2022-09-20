"""Example code for the DEM coregistration chapter (it's made like this to test the syntax). """
#######################
# SECTION: Example data
#######################
import geoutils as gu
import numpy as np

import xdem
from xdem import coreg

# Load the data using xdem and geoutils (could be with rasterio and geopandas instead)
# Load a reference DEM from 2009
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
# Load a moderately well aligned DEM from 1990
tba_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem")).reproject(ref_dem, silent=True)
# Load glacier outlines from 1990. This will act as the unstable ground.
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# This is a boolean numpy 2D array. Note the bitwise not (~) symbol
inlier_mask = ~glacier_outlines.create_mask(ref_dem)

########################
# SECTION: Nuth and Kääb
########################

nuth_kaab = coreg.NuthKaab()
# Fit the data to a suitable x/y/z offset.
nuth_kaab.fit(ref_dem, tba_dem, inlier_mask=inlier_mask)

# Apply the transformation to the data (or any other data)
aligned_dem = nuth_kaab.apply(tba_dem)

####################
# SECTION: Deramping
####################

# Instantiate a 1st order deramping object.
deramp = coreg.Deramp(degree=1)
# Fit the data to a suitable polynomial solution.
deramp.fit(ref_dem, tba_dem, inlier_mask=inlier_mask)

# Apply the transformation to the data (or any other data)
deramped_dem = deramp.apply(tba_dem)

##########################
# SECTION: Bias correction
##########################

vshift_corr = coreg.VerticalShift()
# Note that the transform argument is not needed, since it is a simple vertical correction.
vshift_corr.fit(ref_dem, tba_dem, inlier_mask=inlier_mask)

# Apply the vertical shift to a DEM
corrected_dem = vshift_corr.apply(tba_dem)

# Use median vertical shift instead
vshift_median = coreg.VerticalShift(vshift_func=np.median)

# bias_median.fit(... # etc.

##############
# SECTION: ICP
##############

# Instantiate the object with default parameters
icp = coreg.ICP()
# Fit the data to a suitable transformation.
icp.fit(ref_dem, tba_dem, inlier_mask=inlier_mask)

# Apply the transformation matrix to the data (or any other data)
aligned_dem = icp.apply(tba_dem)

###################
# SECTION: Pipeline
###################

pipeline = coreg.CoregPipeline([coreg.VerticalShift(), coreg.ICP()])

# pipeline.fit(...  # etc.

# This works identically to the syntax above
pipeline2 = coreg.VerticalShift() + coreg.ICP()

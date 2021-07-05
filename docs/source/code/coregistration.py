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
reference_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
# Load a moderately well aligned DEM from 1990
dem_to_be_aligned = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem")).reproject(reference_dem)
# Load glacier outlines from 1990. This will act as the unstable ground.
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

# Prepare the inputs for coregistration.
ref_data = reference_dem.data.squeeze()  # This is a numpy 2D array/masked_array
tba_data = dem_to_be_aligned.data.squeeze()  # This is a numpy 2D array/masked_array
# This is a boolean numpy 2D array. Note the bitwise not (~) symbol
inlier_mask = ~glacier_outlines.create_mask(reference_dem)
transform = reference_dem.transform  # This is a rio.transform.Affine object.

########################
# SECTION: Nuth and Kääb
########################

nuth_kaab = coreg.NuthKaab()
# Fit the data to a suitable x/y/z offset.
nuth_kaab.fit(ref_data, tba_data, transform=transform, inlier_mask=inlier_mask)

# Apply the transformation to the data (or any other data)
aligned_dem = nuth_kaab.apply(tba_data, transform=transform)

####################
# SECTION: Deramping
####################

# Instantiate a 1st order deramping object.
deramp = coreg.Deramp(degree=1)
# Fit the data to a suitable polynomial solution.
deramp.fit(ref_data, tba_data, transform=transform, inlier_mask=inlier_mask)

# Apply the transformation to the data (or any other data)
deramped_dem = deramp.apply(dem_to_be_aligned.data, transform=dem_to_be_aligned.transform)

##########################
# SECTION: Bias correction
##########################

bias_corr = coreg.BiasCorr()
# Note that the transform argument is not needed, since it is a simple vertical correction.
bias_corr.fit(ref_data, tba_data, inlier_mask=inlier_mask, transform=reference_dem.transform)

# Apply the bias to a DEM
corrected_dem = bias_corr.apply(tba_data, transform=dem_to_be_aligned.transform)

# Use median bias instead
bias_median = coreg.BiasCorr(bias_func=np.median)

# bias_median.fit(... # etc.

##############
# SECTION: ICP
##############

# Instantiate the object with default parameters
icp = coreg.ICP()
# Fit the data to a suitable transformation.
icp.fit(ref_data, tba_data, transform=transform, inlier_mask=inlier_mask)

# Apply the transformation matrix to the data (or any other data)
aligned_dem = icp.apply(tba_data, transform=transform)

###################
# SECTION: Pipeline
###################

pipeline = coreg.CoregPipeline([coreg.BiasCorr(), coreg.ICP()])

# pipeline.fit(...  # etc.

# This works identically to the syntax above
pipeline2 = coreg.BiasCorr() + coreg.ICP()

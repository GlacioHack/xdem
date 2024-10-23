"""
Configuring Logging in xDEM
===============================

This example demonstrates how to configure logging in xDEM by using the Nuth and Kääb
(`2011 <https:https://doi.org/10.5194/tc-5-271-2011>`_) coregistration method.
Logging can be customized to various levels, from `DEBUG` for detailed diagnostic output, to `INFO` for general
updates, `WARNING` for potential issues, and `ERROR` or `CRITICAL` for serious problems.

We will demonstrate how to set up logging and show logs while running a typical xDEM function.
"""

import logging

import xdem

# Step 1: Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change this level to DEBUG or WARNING to see different outputs.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("../xdem_example.log"),  # Save logs to a file
        logging.StreamHandler(),  # Print logs to the console
    ],
)

# Step 2: Load example DEMs (Digital Elevation Models) to work with
reference_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_to_be_aligned = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))

# Step 3: Perform Nuth & Kaab coregistration (for alignment of DEMs)
coreg = xdem.coreg.NuthKaab()

# Step 4: Demonstrate logging at various levels
logging.info("Starting Nuth & Kaab coregistration...")
coreg.fit(reference_dem, dem_to_be_aligned)
logging.debug("Coregistration successful. Applying transformation...")

# Apply the coregistration
aligned_dem = coreg.apply(dem_to_be_aligned)

# Output some results
logging.info(f"Coregistration completed. Aligned DEM shape: {aligned_dem.shape}")

# Displaying final logs
logging.debug("Aligned DEM has been computed and saved.")

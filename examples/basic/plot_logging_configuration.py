"""
Configuring verbosity level
===========================

This example demonstrates how to configure verbosity level, or logging, using a coregistration method.
Logging can be customized to various severity levels, from ``DEBUG`` for detailed diagnostic output, to ``INFO`` for
general updates, ``WARNING`` for potential issues, and ``ERROR`` or ``CRITICAL`` for serious problems.

Setting the verbosity to a certain severity level prints all outputs from that level and those above. For instance,
level ``INFO`` also prints warnings, error and critical messages.

See also :ref:`config`.

.. important:: The verbosity level defaults to ``WARNING``, so no ``INFO`` or ``DEBUG`` is printed.
"""

import logging

import xdem

# %%
# We start by configuring the logging level, which can be as simple as specifying we want to print information.
logging.basicConfig(level=logging.INFO)

# %%
# We can change the configuration even more by specifying the format, date, and multiple destinations for the output.
logging.basicConfig(
    level=logging.INFO,  # Change this level to DEBUG or WARNING to see different outputs.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("../xdem_example.log"),  # Save logs to a file
        logging.StreamHandler(),  # Also print logs to the console
    ],
    force=True,  # To re-set from previous logging
)

# %%
# We can now load example files and demonstrate the logging through a functionality, such as coregistration.
reference_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_to_be_aligned = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
coreg = xdem.coreg.NuthKaab()

# %%
# With the ``INFO`` verbosity level defined above, we can follow the iteration with a detailed format, saved to file.
aligned_dem = coreg.fit_and_apply(reference_dem, dem_to_be_aligned)

# %%
# With a more severe verbosity level, there is no output.
logging.basicConfig(level=logging.ERROR, force=True)
aligned_dem = coreg.fit_and_apply(reference_dem, dem_to_be_aligned)

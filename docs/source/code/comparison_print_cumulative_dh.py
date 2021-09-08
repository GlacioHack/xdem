"""Print the Scott Turnerbreen dH series."""
import contextlib
import io
import sys
import xdem

sys.path.insert(0, "code/")  # The base directory is source/, so to find comparison.py, it has to be source/code/

with contextlib.redirect_stdout(io.StringIO()):  # Import the script without printing anything.
    import comparison


dems = xdem.DEMCollection(
    [comparison.dem_1990, comparison.dem_2009, comparison.dem_2060],
    outlines=comparison.outlines,
    reference_dem=comparison.dem_2009
)

dems.subtract_dems()
dems.get_cumulative_series(kind="dh", outlines_filter="NAME == 'Scott Turnerbreen'")

# Create an object that can be printed in the documentation.
scott_series = dems.get_cumulative_series(kind="dh", outlines_filter="NAME == 'Scott Turnerbreen'")

print(scott_series)

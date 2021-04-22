"""Print the Scott Turnerbreen dH series."""
import contextlib
import io
import sys

sys.path.insert(0, "code/")  # The base directory is source/, so to find comparison.py, it has to be source/code/

with contextlib.redirect_stdout(io.StringIO()):  # Import the script without printing anything.
    import comparison

print(comparison.scott_series)

---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: xdem-env
  language: python
  name: xdem
---
# Configuration

xDEM allows to configure the **verbosity level** and the **default behaviour of certain operations on elevation data** (such as
resampling method for reprojection, or pixel interpretation) directly at the package level.

(verbosity)=
## Verbosity level

To configure the verbosity level (or logging) for xDEM, you can utilize Python's built-in `logging` module. This module
has five levels of verbosity that are, in ascending order of severity: `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`.
Setting a level prints output from that level and all other of higher severity. Logging also allows you to specify other aspects,
such as the destination of the output (console, file).

```{important}
**The default verbosity level is `WARNING`, implying that `INFO` and `DEBUG` do not get printed**. Use the basic configuration
as below to setup an `INFO` level.
```

To specify the verbosity level, set up a logging configuration at the start of your script:

```{code-cell} ipython3
import logging

# Basic configuration to simply print info
logging.basicConfig(level=logging.INFO)
```

Optionally, you can specify the logging date, format, and handlers (destinations).

```{code-cell} ipython3

# More advanced configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('app.log'), # Log messages will be saved to this file
        logging.StreamHandler()         # Log messages will also be printed to the console
    ])
```

The above configuration will log messages with a severity level of `INFO` and above, including timestamps, logger names, and
log levels in the output. You can change the logging level as needed.


## Raster–vector–point operations

To change the configuration at the package level regarding operations for rasters, vectors and points, see
[GeoUtils' configuration](https://geoutils.readthedocs.io/en/stable/config.html).

For instance, this allows to define a preferred resampling algorithm used when interpolating and reprojecting
(e.g., bilinear, cubic), or the default behaviour linked to pixel interpretation during point–raster comparison.
These changes will then apply to all your operations in xDEM, such as coregistration.

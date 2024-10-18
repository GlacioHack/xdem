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
# Logging configuration
 TO configure logging for xDEM, you can utilize Python's built-in `logging` module. Begin by setting up the logging
 configuration at the start. This involves specifying the logging level, format, and handlers. For example :
 ```python
import logging

# Configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler('app.log'),  # Log messages will be saved to this file
                        logging.StreamHandler()           # Log messages will also be printed to the console
                    ])
```
This configuration will log messages with a severity level of `INFO` and above, including timestamps, logger names, and
log levels in the output. You can change the logging level to `DEBUG`, `WARNING`, `ERROR` or `CRITICAL` as needed.

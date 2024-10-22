# logging_example.py

"""
This script provides an example for setting up logging in a Python application.
It demonstrates how to configure logging to display messages in the console and save them to a file,
using different logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

Logging configuration allows users to easily control the verbosity of output and log critical information
for debugging and tracking application behavior.

Instructions:
1. Use the `setup_logging()` function to set the logging configuration. You can specify the logging level (e.g., DEBUG, INFO).
   - DEBUG: Provides detailed internal information, useful for debugging.
   - INFO: General messages about the application's operation.
   - WARNING: Indicates potential issues that may not affect functionality.
   - ERROR: Records errors that occur during the execution of the program.
   - CRITICAL: Logs critical failures that require immediate attention.

2. The logs will be saved to a file called 'app.log' and displayed on the console.

3. Example usage of the logger is provided, including basic operations and error handling (division by zero).
   The logger captures different levels of messages as the program runs.

How to run:
- Execute the script, and logs will be generated both in the terminal and saved to 'app.log'.
- Modify the log level in the `setup_logging()` function to control the verbosity.

"""

import logging


def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.

    Parameters:
    log_level (int): The log level for the logger (e.g., logging.DEBUG, logging.INFO).
    """
    # Define the format for logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up basic configuration
    logging.basicConfig(
        level=log_level,  # Set the logging level
        format=log_format,  # Define the format of log messages
        datefmt=date_format,  # Define the date format in log messages
        handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],  # Log to a file  # Log to the console
    )


def main():
    # Set up the logging (adjust the level as needed)
    setup_logging(logging.DEBUG)

    # Log messages at different levels
    logging.debug("This is a debug message, useful for developers to troubleshoot.")
    logging.info("This is an informational message, suitable for general use.")
    logging.warning("This is a warning message, indicating something may go wrong.")
    logging.error("This is an error message, something went wrong!")
    logging.critical("This is a critical message, severe issue encountered.")

    # Example functionality
    def divide(a, b):
        try:
            result = a / b
            logging.info("Division successful: %d / %d = %f", a, b, result)
            return result
        except ZeroDivisionError:
            logging.error("Error: Attempted division by zero")
            return None

    # Run some example calculations
    divide(10, 2)
    divide(10, 0)


if __name__ == "__main__":
    main()

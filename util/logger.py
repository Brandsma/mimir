"""
This module contains the logger functions

Levels are used for identifying the severity of an event.
There are six logging levels:

    CRITICAL
    ERROR
    WARNING
    INFO
    DEBUG
    NOTSET

If the logging level is set to WARNING, all WARNING, ERROR, and CRITICAL
messages are written to the log file or console.
If it is set to ERROR, only ERROR and CRITICAL messages are logged.

"""

import logging

FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s"


def set_colour_coded_levels():
    """ Set custom colours for the several debug levels. This is somewhat hacky """
    logging.addLevelName(
        logging.DEBUG, "\033[0;24m%s\033[1;0m" % logging.getLevelName(logging.DEBUG)
    )
    logging.addLevelName(
        logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO)
    )
    logging.addLevelName(
        logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
    )
    logging.addLevelName(
        logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR)
    )
    logging.addLevelName(
        logging.CRITICAL,
        "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL),
    )


def setup_logger(name, level=logging.INFO):
    """
    *setup_logger* returns a logger with specific formatting options

    name should almost always be __name__
    to represent the name of the module it is called from

    This function should be called once globally in a module
    """
    set_colour_coded_levels()

    # Create a custom logger
    log = logging.getLogger(name)

    # Only the messages of level and above will be logged
    log.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    c_format = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(c_format)

    # Apply the handler to the logger
    log.addHandler(c_handler)

    log.info("Logger %s has been initialized", name)

    return log


# This is for testing/showcase purposes
if __name__ == "__main__":
    log = setup_logger(__name__, level=logging.DEBUG)
    log.debug("This is some handy debug information")
    log.info("The answer to life, universe and everything is 42")
    log.warning("Wait, where's Barry?")
    log.error("Thanks for all the fish!")
    log.critical("Time to panic.")

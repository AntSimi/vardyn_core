import logging
import os

# Disable HDF5 file locking BEFORE importing xarray/netCDF4. On shared
# filesystems (NFS/Lustre) concurrent reads from sibling subprocesses can
# raise "NetCDF: Not a valid ID" otherwise.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def start_logger():
    FORMAT_LOG = "%(levelname)-8s %(asctime)s %(module)s.%(funcName)s : %(message)s"
    logger = logging.getLogger("vardyn")
    if len(logger.handlers) == 0:
        # set up logging to CONSOLE
        console = logging.StreamHandler()
        console.setFormatter(ColoredFormatter(FORMAT_LOG))
        # add the handler to the root logger
        logger.addHandler(console)
    return logger


class ColoredFormatter(logging.Formatter):
    COLOR_LEVEL = dict(
        CRITICAL="\037[37;41m",
        ERROR="\033[31;47m",
        WARNING="\033[30;47m",
        INFO="\033[36m",
        DEBUG="\033[34m",
    )

    def __init__(self, message):
        super().__init__(message)

    def format(self, record):
        color = self.COLOR_LEVEL.get(record.levelname, "")
        color_reset = "\033[0m"
        model = color + "%s" + color_reset
        record.msg = model % record.msg
        record.funcName = model % record.funcName
        record.module = model % record.module
        record.levelname = model % record.levelname
        return super().format(record)

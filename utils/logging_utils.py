import logging
from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logging.Logger object with configured
    RichHandler for cool logging
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        rich_handler = RichHandler(
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        # formatter = logging._defaultFormatter
        formatter = logging.Formatter(
            "%(message)s",
            "%H:%M:%S")
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)
        logger.propagate = False

    return logger


def set_log_level(verbosity: bool):
    if verbosity:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # for name in logging.root.manager.loggerDict:
    #     logging.getLogger(name).setLevel(level)
    logging.getLogger().setLevel(level)

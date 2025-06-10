from threading import Lock
from .logging_iface import DataLoggerProtocol
from .rerun_logging import DataLoggerRR
from utils.config_utils import Configuration, DataLoggerType

datalogger_available = {
    DataLoggerType.rerun: DataLoggerRR
}

_lock = Lock()
_datalogger: DataLoggerProtocol | None = None


def get_datalogger(cfg: Configuration):
    global _datalogger

    if _datalogger is None:
        with _lock:
            _datalogger = datalogger_available[cfg.logging.logger_type](cfg)
    return _datalogger

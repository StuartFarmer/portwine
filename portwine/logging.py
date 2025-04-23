# portwine/logging.py
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler


class Logger:
    """
    Custom logger that outputs styled logs to the console using Rich
    and optionally writes to a rotating file handler.
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Path = None,
        rotate: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """
        Initialize and configure the logger.

        :param name: Name of the logger (usually __name__).
        :param level: Logging level.
        :param log_file: Path to the log file; if provided, file handler is added.
        :param rotate: Whether to use a rotating file handler.
        :param max_bytes: Maximum size of a log file before rotation (in bytes).
        :param backup_count: Number of rotated backup files to keep.
        """
        # Create or get the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # Prevent logs from propagating to the root logger twice
        self.logger.propagate = False

        # Console handler with Rich
        console_handler = RichHandler(
            level=level,
            show_time=True,
            markup=True,
            rich_tracebacks=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        console_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            if rotate:
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)

            file_formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def get(self) -> logging.Logger:
        """
        Return the configured standard logger instance.
        """
        return self.logger

    @classmethod
    def create(
        cls,
        name: str,
        level: int = logging.INFO,
        log_file: Path = None,
        rotate: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Convenience method to configure and return a logger in one step.
        """
        return cls(name, level, log_file, rotate, max_bytes, backup_count).get() 
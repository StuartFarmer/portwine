# portwine/logging.py
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich import print as rprint
from rich.table import Table
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from typing import Dict, List
from portwine.brokers.base import Order

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
        propagate: bool = False,
    ):
        """
        Initialize and configure the logger.

        :param name: Name of the logger (usually __name__).
        :param level: Logging level.
        :param log_file: Path to the log file; if provided, file handler is added.
        :param rotate: Whether to use a rotating file handler.
        :param max_bytes: Maximum size of a log file before rotation (in bytes).
        :param backup_count: Number of rotated backup files to keep.
        :param propagate: Whether to propagate logs to parent loggers (default False).
        """
        # Create or get the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # Allow control over log propagation
        self.logger.propagate = propagate

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
        propagate: bool = False,
    ) -> logging.Logger:
        """
        Convenience method to configure and return a logger in one step.
        """
        return cls(name, level, log_file, rotate, max_bytes, backup_count, propagate).get()

# Top-level rich-logging helpers
def log_position_table(logger: logging.Logger, current_positions: Dict[str, float], target_positions: Dict[str, float]) -> None:
    """Pretty-print position changes as a Rich table"""
    table = Table(title="Position Changes")
    table.add_column("Ticker")
    table.add_column("Current", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Change", justify="right")
    for t in sorted(set(current_positions) | set(target_positions)):
        curr = current_positions.get(t, 0)
        tgt = target_positions.get(t, 0)
        table.add_row(t, f"{curr:.4f}", f"{tgt:.4f}", f"{tgt-curr:.4f}")
    logger.info("Position changes:")
    rprint(table)

def log_weight_table(logger: logging.Logger, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> None:
    """Pretty-print weight changes as a Rich table"""
    table = Table(title="Weight Changes")
    table.add_column("Ticker")
    table.add_column("Current Wt", justify="right")
    table.add_column("Target Wt", justify="right")
    table.add_column("Delta Wt", justify="right")
    for t in sorted(set(current_weights) | set(target_weights)):
        cw = current_weights.get(t, 0)
        tw = target_weights.get(t, 0)
        table.add_row(t, f"{cw:.2%}", f"{tw:.2%}", f"{(tw-cw):.2%}")
    logger.info("Weight changes:")
    rprint(table)

def log_order_table(logger: logging.Logger, orders: List[Order]) -> None:
    """Pretty-print orders to execute as a Rich table"""
    table = Table(title="Orders to Execute")
    table.add_column("Ticker")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Type")
    table.add_column("TIF")
    table.add_column("Price", justify="right")
    for o in orders:
        table.add_row(o.ticker, o.side, str(int(o.quantity)), o.order_type, o.time_in_force, f"{o.average_price:.2f}")
    logger.info("Orders to execute:")
    rprint(table)

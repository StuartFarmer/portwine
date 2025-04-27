#!/usr/bin/env python3
"""
A demo script showcasing the custom Logger with Rich:
- Colored messages at different levels
- Markup in messages
- Progress bar and spinner
- Rich exception tracebacks
- Logging tables and panels
- File output in `logs/demo.log`
"""
import time
import logging
from pathlib import Path

from portwine.logging import Logger
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel


def main():
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "demo.log"

    # Create logger (console + file)
    logger = Logger.create(
        name=__name__,
        level=logging.DEBUG,
        log_file=log_file,
        rotate=True,
        max_bytes=1_000_000,
        backup_count=2,
    )

    # Demonstrate each standard level
    logger.debug("[dim]Debug message (verbose, for developers)[/dim]")
    logger.info("[cyan]Info message (general info)[/cyan]")
    logger.warning("[yellow]Warning message (something might be off)[/yellow]")
    logger.error("[red]Error message (something went wrong)[/red]")
    logger.critical("[bold red]Critical message (serious failure)[/bold red]")

    # Show a progress bar
    logger.info("Starting a demo progress bar…")
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Processing items...", total=100)
        for _ in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)
    logger.info("Progress complete!")

    # Demonstrate a rich exception
    try:
        1 / 0
    except Exception:
        logger.exception("Caught an exception with rich traceback")

    # Log a table
    table = Table(title="My Cool Table")
    table.add_column("Name", style="bold magenta")
    table.add_column("Quantity", justify="right", style="green")
    table.add_row("Apples", "10")
    table.add_row("Bananas", "20")
    table.add_row("Cherries", "5")
    logger.info("Rendering table below:")
    rprint(table)
    logger.info("Table rendered.")

    # Log a panel
    panel = Panel.fit(
        "Hello from [bold green]Rich[/bold green]!",
        title="Greeting",
        border_style="blue",
    )
    logger.info("Rendering panel below:")
    rprint(panel)
    logger.info("Panel rendered.")

    # Use rich.print for an extra demonstration
    rprint("[bold underline]Demo complete![/bold underline]")
    rprint(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    main() 
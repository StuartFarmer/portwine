import unittest
import logging
import tempfile
from pathlib import Path
from portwine.logger import Logger
from logging.handlers import RotatingFileHandler
from logging import FileHandler

class TestLogger(unittest.TestCase):
    def test_create_console_only(self):
        """
        Logger.create without log_file should only attach a RichHandler at the specified level.
        """
        name = "test_console"
        level = logging.WARNING
        logger = Logger.create(name, level=level)

        # Check logger instance and properties
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, name)
        self.assertEqual(logger.level, level)
        self.assertFalse(logger.propagate)

        # Only one handler (RichHandler)
        handlers = logger.handlers
        self.assertEqual(len(handlers), 1)
        handler = handlers[0]
        from rich.logging import RichHandler
        self.assertIsInstance(handler, RichHandler)
        self.assertEqual(handler.level, level)

    def test_create_file_handler_rotating(self):
        """
        Logger.create with rotate=True should attach a RotatingFileHandler.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "app.log"
            level = logging.DEBUG
            max_bytes = 2048
            backup_count = 3
            logger = Logger.create(
                "test_file_rot",
                level=level,
                log_file=log_path,
                rotate=True,
                max_bytes=max_bytes,
                backup_count=backup_count,
            )

            # Expect two handlers: RichHandler + RotatingFileHandler
            self.assertEqual(len(logger.handlers), 2)

            # Identify the rotating file handler and check its settings
            rot_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
            self.assertEqual(len(rot_handlers), 1)
            rh = rot_handlers[0]
            self.assertEqual(rh.maxBytes, max_bytes)
            self.assertEqual(rh.backupCount, backup_count)

            # Emit logs and verify file output
            logger.info("info msg")
            logger.debug("debug msg")
            # Flush and read
            rh.flush()
            content = log_path.read_text()
            self.assertIn("info msg", content)
            self.assertIn("debug msg", content)

            # Each line should follow the format 'YYYY-MM-DD HH:MM:SS | name | LEVEL | message'
            lines = [ln for ln in content.splitlines() if ln]
            for ln in lines:
                parts = [p.strip() for p in ln.split("|")]
                # Check 4 segments: timestamp, logger name, level, message
                self.assertEqual(len(parts), 4)
                self.assertRegex(parts[0], r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
                self.assertEqual(parts[1], "test_file_rot")
                self.assertIn(parts[2], ["INFO", "DEBUG"])

    def test_create_file_handler_no_rotate(self):
        """
        Logger.create with rotate=False should attach a FileHandler (not rotating).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "app.log"
            logger = Logger.create(
                "test_file_nr", level=logging.ERROR, log_file=log_path, rotate=False
            )

            # Expect two handlers: RichHandler + FileHandler
            self.assertEqual(len(logger.handlers), 2)

            file_handlers = [h for h in logger.handlers if isinstance(h, FileHandler)]
            self.assertEqual(len(file_handlers), 1)

            fh = file_handlers[0]
            logger.error("error occurred")
            fh.flush()
            content = log_path.read_text()
            self.assertIn("error occurred", content)

            # Rotating backups should not exist
            backups = list(Path(tmpdir).glob("app.log.*"))
            self.assertEqual(backups, [])

    def test_logging_levels(self):
        """
        Verify that logging respects the logger's level (INFO should suppress DEBUG).
        """
        name = "test_levels"
        logger = Logger.create(name, level=logging.INFO, propagate=True)

        # Capture logs emitted through the logger itself
        with self.assertLogs(name, level="INFO") as cm:
            logger.debug("this should not appear")
            logger.info("info ok")
            logger.warning("warn ok")

        # Only two messages at INFO and WARNING should be captured
        self.assertEqual(len(cm.output), 2)
        self.assertIn("INFO:test_levels:info ok", cm.output[0])
        self.assertIn("WARNING:test_levels:warn ok", cm.output[1])
        # DEBUG message must not be present
        self.assertNotIn("this should not appear", "\n".join(cm.output))

if __name__ == "__main__":
    unittest.main() 
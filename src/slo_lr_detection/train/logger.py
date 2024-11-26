import logging
import json
import os


class JSONFileHandler(logging.Handler):
    """
    Custom logging handler that writes log records to a JSON file.

    Args:
        filename (str): Path to the JSON file where log records will be stored.
    """

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._initialize_file()

    def _initialize_file(self):
        """Initialize the JSON file if it doesn't exist."""
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                json.dump([], f)  # Create an empty list in the JSON file

    def emit(self, record):
        """
        Write a log record to the JSON file.

        Args:
            record (logging.LogRecord): The log record to write to the file.
        """
        log_entry = self.format(record)
        with open(self.filename, "r+") as f:
            data = json.load(f)
            data.append(json.loads(log_entry))  # Append the new log entry
            f.seek(0)  # Rewind file pointer to the beginning
            json.dump(data, f, indent=4)  # Overwrite with the updated data


class CustomJSONFormatter(logging.Formatter):
    """
    Custom formatter for logging records in JSON format.

    Args:
        **kwargs: Additional fields to include in the log record.
    """

    def __init__(self, **kwargs):
        self.extra_fields = kwargs
        log_format = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"'
        )

        # If there are extra fields, include them in the format string
        if kwargs:
            fields = ", ".join([f'"{k}": "%({k})s"' for k in kwargs.keys()])
            log_format += f", {fields}"

        log_format += "}"
        super().__init__(log_format)

    def format(self, record):
        """
        Add extra fields to the log record and format it as JSON.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record as a JSON string.
        """
        for key, value in self.extra_fields.items():
            setattr(record, key, value)
        return super().format(record)


def setup_logger(name, filename, level="INFO", **kwargs):
    """
    Setup a logger that outputs log records to a JSON file.

    Args:
        name (str): The name of the logger.
        filename (str): The path to the JSON file where logs will be saved.
        level (str): The logging level (default is "INFO").
        **kwargs: Additional fields to include in the log record.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Map string log level to logging module constants
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Get the log level from the string, default to INFO if the level is invalid
    log_level = log_levels.get(level.upper(), logging.INFO)

    # Create and configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    logger.handlers.clear()

    # Create and configure the JSON file handler
    handler = JSONFileHandler(filename)
    formatter = CustomJSONFormatter(**kwargs)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    # Example usage
    logger = setup_logger(name="my_logger", filename=r"./log.json", level="ERROR")
    logger.info("This is an info message.")  # Will not be logged due to log level
    logger.error("This is an error message.")  # Will be logged

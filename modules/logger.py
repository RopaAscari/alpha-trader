import logging

class AppLogger:
    def __init__(self, logger_name, log_level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # Check if console handler has already been added
        handlers = self.logger.handlers
        if not any(isinstance(h, logging.StreamHandler) for h in handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            formatter = ColorFormatter('[%(name)s] [%(asctime)s] [%(levelname)s] - %(message)s')
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        'debug': '\033[36m',  # cyan
        'info': '\033[32m',   # green
        'warning': '\033[33m',  # yellow
        'error': '\033[31m',   # red
        'critical': '\033[1;31m',  # bold red
    }
    RESET_CODE = '\033[0m'

    def format(self, record):
        record.levelname = record.levelname.lower()
        message = super().format(record)
        color_code = self.COLOR_CODES.get(record.levelname, '')
        return f"{color_code}{message}{self.RESET_CODE}"
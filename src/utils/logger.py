import logging

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def get_console_logger(level='info'):
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    if not logger.handlers:
        # Add handler to log to the console
        sh = logging.StreamHandler()
        sh.setLevel(level.upper())
        sh.setFormatter(log_formatter)

        logger.addHandler(sh)

    return logger


def add_file_handler(logger, file_path, level='info', exclusive=False):
    if exclusive:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)


def get_file_logger(file_path, level='info', propagate=False):
    logger = logging.getLogger()
    logger.propagate = propagate
    logger.setLevel(getattr(logging, level.upper()))

    assert not logger.handlers
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


log = get_console_logger()

import logging

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def get_console_logger(level='info'):
    logger = logging.getLogger()
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)
    elif level == 'error':
        logger.setLevel(logging.ERROR)
    elif level == 'critical':
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.NOTSET)

    if not logger.handlers:
        # Add handler to log to the console
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(log_formatter)

        logger.addHandler(sh)

    return logger

log = get_console_logger()

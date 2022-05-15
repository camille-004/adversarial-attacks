# type: ignore
import logging
import os
import sys

LOG_DIR_NAME = "logs"


def setup_module_level_logger(
    module_name: str, is_debug: bool = True, file_name: str = None
) -> logging.Logger:
    """
    Set up a module-level logger.

    :param module_name: Module which will use logger.
    :param is_debug: Set to level to DEBUG if True.
    :param file_name: File to which to save logs.
    :return:
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG if is_debug else logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if file_name:
        if not os.path.exists(LOG_DIR_NAME):
            os.makedirs(LOG_DIR_NAME)
        fh = logging.FileHandler(os.path.join(LOG_DIR_NAME, file_name))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

import logging
from rich.logging import RichHandler


def setup_logging(fname=None):
    # log = logging.getLogger(__name__)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler(show_level=True,show_path=False)])
    logger = logging.getLogger('durip')
    logger.setLevel(logging.DEBUG)
    if fname is not None:
        fh = logging.FileHandler(fname, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    return logger
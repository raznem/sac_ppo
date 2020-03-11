import logging
import sys
from datetime import datetime
from pathlib import Path


def get_logger():
    file = sys.argv[0]
    path = Path(file)
    name = path.name.split(".py")[0]
    directory = path.parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / (name + "_" + datetime.now().isoformat() + ".log")
    logging.basicConfig(
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(processName)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return logger

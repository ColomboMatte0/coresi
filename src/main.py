import argparse
import logging
import sys
from pathlib import Path

import yaml

from camera import setup_cameras
from data import read_data_file

parser = argparse.ArgumentParser(description="CORESI")

parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Enable debug output",
)
parser.add_argument(
    "-c",
    "--config",
    default="config.yaml",
    help="Path to the configuration file",
    type=Path,
)

args = parser.parse_args()

file_handler = logging.FileHandler(filename="coresi.log", mode="w")
stdout_handler = logging.StreamHandler()
handlers = (file_handler, stdout_handler)

logging.basicConfig(
    level=logging.INFO if not args.verbose else logging.DEBUG,
    format="[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger("CORESI")

logger.info(f"Reading configuration file {args.config}")
try:
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)
except IOError as e:
    logger.fatal(f"Failed to open the configuration file: {e}")
    sys.exit(1)

# Setup the cameras' list according to their characteristics
cameras = setup_cameras(config["cameras"])
logger.info(f"Processing {config['data_file']}")

# Process events from the data file and associate them with the cameras
events = read_data_file(
    Path(config["data_file"]),
    n_events=30,
    E0=config["E0"],
    cameras=cameras,
    start_position=0,
)

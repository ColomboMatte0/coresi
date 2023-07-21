import argparse
import logging
import sys
from pathlib import Path
import time

import yaml

from camera import setup_cameras
from data import read_data_file
from image import Image
from mlem import LM_MLEM

parser = argparse.ArgumentParser(description="CORESI")

start = time.time()

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
    n_events=1,
    E0=config["E0"],
    cameras=cameras,
    # Needed to remove events with energy outside of a given range
    energy_range=config["energy_range"],
    start_position=0,
    remove_out_of_range_energies=config["remove_out_of_range_energies"],
)

logger.info(f"Took {time.time() - start} ms to read the data")

# Reinitialize the timer for MLEM
start = time.time()

logger.info("Doing MLEM")
# Image share the same properties as system matrix line (i.e. Ti), pass it by copy
mlem = LM_MLEM(config["lm_mlem"], config["volume"], cameras, events)
result = mlem.run(config["lm_mlem"]["last_iter"], config["lm_mlem"]["first_iter"])
# image = mlem.SM_angular_thickness(events[0])

logger.info(f"Took {time.time() - start} ms for MLEM")

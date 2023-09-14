import argparse
from pathlib import Path

import numpy as np
import yaml

from image import Image

parser = argparse.ArgumentParser(description="CORESI")

parser.add_argument("-i", "--image", type=Path, required=True)
parser.add_argument(
    "-c",
    "--config",
    default="config.yaml",
    help="Path to the configuration file",
    type=Path,
)
args = parser.parse_args()

with open(args.config, "r") as fh:
    config = yaml.safe_load(fh)

image = Image(config["volume"])
image.values = np.load(args.image)
image.display_z()

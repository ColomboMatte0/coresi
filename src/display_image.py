import argparse
from pathlib import Path

import torch
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

image = Image(len(config["E0"]), config["volume"])
image.values = torch.load(args.image, map_location=torch.device("cpu"))

for e in range(image.values.shape[0]):
    image.display_z(energy=e, title=f" {str(config['E0'][e])} keV")

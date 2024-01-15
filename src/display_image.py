import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
parser.add_argument(
    "--cpp",
    action=argparse.BooleanOptionalAction,
    help="Use this if the file comes from the C++ version of CORESI",
    type=bool,
)
args = parser.parse_args()

with open(args.config, "r") as fh:
    config = yaml.safe_load(fh)

image = Image(len(config["E0"]), config["volume"])

if args.cpp:
    image.values = torch.from_numpy(
        np.fromfile(args.image).reshape(image.values.shape).transpose(-4, -2, -3, -1)
    )
else:
    image.values = torch.load(args.image, map_location=torch.device("cpu"))

for e in range(image.values.shape[0]):
    image.display_z(
        energy=e, title=f" {str(config['E0'][e])} keV" + (" CPP" if args.cpp else "")
    )

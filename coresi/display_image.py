# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import git
import numpy as np
import torch
import yaml

from coresi.image import Image

parser = argparse.ArgumentParser(
    description="CORESI - image display",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
repo = git.Repo(search_parent_directories=True)
commit = repo.git.rev_parse("HEAD", short=True)

parser.add_argument(
    "-f", "--file", type=Path, required=True, help="File path to the volume to display"
)
parser.add_argument(
    "-c",
    "--config",
    default="config.yaml",
    help="Path to the configuration file",
    type=Path,
)

parser.add_argument(
    "-s",
    "--slice",
    default=0,
    help="Slice number",
    type=int,
)
parser.add_argument(
    "-a",
    "--axis",
    default="z",
    help="Axis x, y or z",
    type=str,
)
parser.add_argument(
    "--cpp",
    action=argparse.BooleanOptionalAction,
    help="Use this if the file comes from the C++ version of CORESI",
    type=bool,
)
args = parser.parse_args()


def display():
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    image = Image(len(config["E0"]), config["volume"])

    if args.cpp:
        image.values = torch.from_numpy(
            np.fromfile(args.file)
            .reshape(image.values.shape)
            .transpose(-4, -2, -3, -1)
        )
    else:
        image.values = torch.load(args.file, map_location=torch.device("cpu"))

    for e in range(image.values.shape[0]):
        if args.axis == "z":
            image.display_z(
                energy=e,
                title=f" {str(config['E0'][e])} keV" + (" CPP" if args.cpp else ""),
                slice=args.slice,
            )
        if args.axis == "x":
            image.display_x(
                energy=e,
                title=f" {str(config['E0'][e])} keV" + (" CPP" if args.cpp else ""),
                slice=args.slice,
            )
        if args.axis == "y":
            image.display_y(
                energy=e,
                title=f" {str(config['E0'][e])} keV" + (" CPP" if args.cpp else ""),
                slice=args.slice,
            )


if __name__ == "__main__":
    display()
